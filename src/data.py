"""
data.py — Load and clean the Kestrel warranty claims data.

Handles, in order, everything found during Phase 1 profiling:
  1. Parse timestamps.
  2. Sanitize claim_description (strip suspected prompt-injection payloads
     before any text-derived feature is built from it).
  3. Deduplicate resubmitted claims (keep earliest submission).
  4. Split into decided (usable for supervised learning) vs undecided
     (still in the investigation queue — no ground truth, excluded).
  5. Normalize product_serial.
  6. Join partners.csv and products.csv.

Nothing here computes cross-row statistics (partner fraud rate, etc.) —
that is deliberately kept in features.py, where the point-in-time
discipline lives. data.py only cleans each row using information that
row already carries.
"""
from __future__ import annotations

import re
import pandas as pd

# Normal claim_description length in this dataset tops out at 24 characters
# (verified: max legitimate length in both train and test is 24). Anything
# meaningfully longer is, by construction, not a fault description anymore.
# We don't pattern-match specific injected phrases (that overfits to the
# wording we've already seen) — instead we truncate anything abnormally
# long, which neutralizes an injected payload regardless of its wording.
DESCRIPTION_SAFE_LEN = 30


def sanitize_claim_description(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Returns (sanitized_series, flagged_mask).
    Anything longer than DESCRIPTION_SAFE_LEN is truncated; the full
    original text is never used downstream. flagged_mask marks rows that
    were truncated, for the audit log — it is NOT a model feature.
    """
    s = series.fillna("").astype(str)
    flagged = s.str.len() > DESCRIPTION_SAFE_LEN
    sanitized = s.where(~flagged, s.str.slice(0, DESCRIPTION_SAFE_LEN).str.strip())
    return sanitized, flagged


def normalize_serial(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str).str.upper()
    s = s.str.replace(r"[-\s]", "", regex=True)
    return s.str.strip()


def load_raw(train_path: str, test_path: str, partners_path: str, products_path: str):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    partners = pd.read_csv(partners_path)
    products = pd.read_csv(products_path)

    train["submitted_at"] = pd.to_datetime(train["submitted_at"])
    test["submitted_at"] = pd.to_datetime(test["submitted_at"])
    partners["onboarded_date"] = pd.to_datetime(partners["onboarded_date"])

    return train, test, partners, products


def _clean_one(df: pd.DataFrame, partners: pd.DataFrame, products: pd.DataFrame,
                is_train: bool, audit: dict) -> pd.DataFrame:
    df = df.copy()

    # --- 1. sanitize free text before it can inform any feature ---
    df["claim_description_clean"], desc_flagged = sanitize_claim_description(df["claim_description"])
    if desc_flagged.any():
        audit.setdefault("sanitized_description_claim_ids", []).extend(
            df.loc[desc_flagged, "claim_id"].tolist()
        )
    df["inspector_note_clean"], note_flagged = sanitize_claim_description(
        df["inspector_note"].fillna("")
    )
    # inspector_note is legitimately longer sometimes (free text from a human
    # inspector) so we use a looser cap for it — just guard against something
    # absurd/adversarial, not the normal range of a genuine note.
    df["inspector_note_clean"] = df["inspector_note"].fillna("")
    long_note = df["inspector_note_clean"].str.len() > 200
    if long_note.any():
        audit.setdefault("sanitized_inspector_note_claim_ids", []).extend(
            df.loc[long_note, "claim_id"].tolist()
        )
        df.loc[long_note, "inspector_note_clean"] = df.loc[long_note, "inspector_note_clean"].str.slice(0, 200)

    # --- 2. normalize serial ---
    df["serial_norm"] = normalize_serial(df["product_serial"])

    # --- 3. dedup resubmissions: keep earliest submission per claim_id ---
    before = len(df)
    df = df.sort_values("submitted_at").drop_duplicates(subset="claim_id", keep="first")
    audit["duplicates_dropped"] = audit.get("duplicates_dropped", 0) + (before - len(df))

    # --- 4. split decided / undecided (train only; test has no is_fraud) ---
    if is_train:
        undecided_mask = df["is_fraud"].isna()
        audit["undecided_dropped"] = int(undecided_mask.sum())
        audit["undecided_claim_ids"] = df.loc[undecided_mask, "claim_id"].tolist()
        df = df.loc[~undecided_mask].copy()
        df["is_fraud"] = df["is_fraud"].astype(int)

    # --- 5. join partners + products ---
    df = df.merge(partners, on="partner_id", how="left", validate="m:1")
    df = df.merge(products, on="sku", how="left", validate="m:1")

    # --- 6. simple row-local derived fields (no cross-row stats here) ---
    df["tenure_days"] = (df["submitted_at"] - df["onboarded_date"]).dt.days
    df["is_small_claim"] = df["claim_amount_inr"] < 2000
    df["post_inspection_policy_change"] = df["submitted_at"] >= pd.Timestamp("2026-05-01")
    df["photo_attached_bin"] = (df["photo_attached"] == "Y").astype(int)
    df["partner_inspected_bin"] = (df["partner_inspected"] == "Y").astype(int)
    df["claim_to_price_ratio"] = df["claim_amount_inr"] / df["list_price_inr"]
    df["desc_len"] = df["claim_description_clean"].str.len()

    return df.sort_values("submitted_at").reset_index(drop=True)


def load_and_clean(train_path: str, test_path: str, partners_path: str, products_path: str):
    """
    Returns (train_clean, test_clean, audit_log).
    train_clean contains only decided, deduplicated claims, sorted by time.
    test_clean is deduplicated (defensive — none expected) and joined,
    with no label.
    """
    train_raw, test_raw, partners, products = load_raw(
        train_path, test_path, partners_path, products_path
    )
    audit: dict = {}

    train_clean = _clean_one(train_raw, partners, products, is_train=True, audit=audit)

    test_audit: dict = {}
    test_clean = _clean_one(test_raw, partners, products, is_train=False, audit=test_audit)
    audit["test_duplicates_dropped"] = test_audit.get("duplicates_dropped", 0)

    # Sanity checks that should never fail on this data pack; fail loudly if they do.
    assert train_clean["partner_id"].notna().all(), "unmatched partner_id in train after join"
    assert train_clean["family"].notna().all(), "unmatched sku in train after join"
    assert test_clean["partner_id"].notna().all(), "unmatched partner_id in test after join"
    assert test_clean["family"].notna().all(), "unmatched sku in test after join"
    assert train_clean["claim_id"].is_unique
    assert test_clean["claim_id"].is_unique
    assert set(train_clean["claim_id"]).isdisjoint(set(test_clean["claim_id"])), \
        "train/test claim_id overlap — temporal split is compromised"

    return train_clean, test_clean, audit


if __name__ == "__main__":
    import config

    config.require_data_files()
    train_clean, test_clean, audit = load_and_clean(
        str(config.TRAIN_PATH), str(config.TEST_PATH),
        str(config.PARTNERS_PATH), str(config.PRODUCTS_PATH),
    )
    print("train_clean:", train_clean.shape, "fraud rate: {:.4f}%".format(
        100 * train_clean["is_fraud"].mean()))
    print("test_clean:", test_clean.shape)
    print("audit summary:")
    print(" duplicates_dropped (train):", audit["duplicates_dropped"])
    print(" duplicates_dropped (test):", audit["test_duplicates_dropped"])
    print(" undecided_dropped:", audit["undecided_dropped"])
    print(" sanitized_description_claim_ids:", audit.get("sanitized_description_claim_ids", []))
