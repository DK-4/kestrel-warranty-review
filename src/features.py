"""
features.py — Point-in-time-safe feature engineering.

The rule for every group-level statistic in this file: a claim's feature
value may only be built from OTHER claims whose outcome was knowable
strictly BEFORE that claim's own submitted_at. A claim never sees its own
label, and never sees any claim submitted at the same time or later.

For TRAIN rows this means an expanding (cumulative, shifted-by-one)
window within the sorted claim history.

For TEST rows, every test claim (Jul-Sep 2026) is strictly after every
train claim (Apr 2025-Jun 2026), so each test claim's feature value is
simply the FINAL accumulated statistic from the full train history for
its group (partner / partner_type). Test claims never inform each
other's features either — we don't get to peek at other test outcomes,
because we don't have them.

GLOBAL_FRAUD_RATE and the smoothing constant are fit once on train and
reused as fixed constants for test — never refit on data that includes
test.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

SMOOTHING_ALPHA = 20  # pseudo-count: how many "prior" pseudo-claims at the
                       # global rate we blend in before trusting a group's
                       # own observed rate. Larger = more shrinkage for
                       # sparse groups.


def _expanding_group_stats(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    df must be sorted by submitted_at ascending, decided train rows only,
    with an integer is_fraud column. Returns two new columns:
      <group_col>_prior_n     : # prior decided claims in this group
      <group_col>_prior_fraud : # of those that were fraud
    computed via a strictly-prior expanding window (current row excluded).
    """
    grp = df.groupby(group_col)["is_fraud"]
    cum_n = grp.cumcount()  # count of rows before this one in the group (0-indexed -> already excludes self)
    cum_fraud = grp.cumsum() - df["is_fraud"]  # cumsum includes self; subtract self out
    out = pd.DataFrame({
        f"{group_col}_prior_n": cum_n.values,
        f"{group_col}_prior_fraud": cum_fraud.values,
    }, index=df.index)
    return out


def _final_group_stats(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Full-history (end-of-train) totals per group, for attaching to test."""
    return df.groupby(group_col)["is_fraud"].agg(n="count", fraud="sum")


def _smoothed_rate(prior_fraud, prior_n, global_rate, alpha=SMOOTHING_ALPHA):
    return (prior_fraud + alpha * global_rate) / (prior_n + alpha)


def build_features(train_clean: pd.DataFrame, test_clean: pd.DataFrame):
    """
    Returns (train_feat, test_feat, feature_cols, meta) — does not mutate inputs.
    train_clean/test_clean are the outputs of data.load_and_clean().
    """
    train = train_clean.sort_values("submitted_at").reset_index(drop=True).copy()
    test = test_clean.copy()

    global_rate = train["is_fraud"].mean()  # fit on train only, fixed constant thereafter

    feature_cols = []

    # ---------- partner_id level ----------
    stats = _expanding_group_stats(train, "partner_id")
    train = pd.concat([train, stats], axis=1)
    train["partner_id_hist_rate"] = _smoothed_rate(
        train["partner_id_prior_fraud"], train["partner_id_prior_n"], global_rate
    )

    partner_final = _final_group_stats(train, "partner_id")
    test = test.merge(
        partner_final.rename(columns={"n": "partner_id_prior_n", "fraud": "partner_id_prior_fraud"}),
        left_on="partner_id", right_index=True, how="left",
    )
    # cold-start partners (14 of them): no train history at all -> 0/0
    test["partner_id_prior_n"] = test["partner_id_prior_n"].fillna(0)
    test["partner_id_prior_fraud"] = test["partner_id_prior_fraud"].fillna(0)
    test["partner_id_hist_rate"] = _smoothed_rate(
        test["partner_id_prior_fraud"], test["partner_id_prior_n"], global_rate
    )
    feature_cols += ["partner_id_hist_rate", "partner_id_prior_n"]

    # ---------- partner_type level (fallback signal, also useful on its own) ----------
    stats = _expanding_group_stats(train, "partner_type")
    stats = stats.rename(columns={
        "partner_type_prior_n": "ptype_prior_n", "partner_type_prior_fraud": "ptype_prior_fraud"
    })
    train = pd.concat([train, stats], axis=1)
    train["ptype_hist_rate"] = _smoothed_rate(train["ptype_prior_fraud"], train["ptype_prior_n"], global_rate)

    ptype_final = _final_group_stats(train, "partner_type")
    test = test.merge(
        ptype_final.rename(columns={"n": "ptype_prior_n", "fraud": "ptype_prior_fraud"}),
        left_on="partner_type", right_index=True, how="left",
    )
    test["ptype_hist_rate"] = _smoothed_rate(test["ptype_prior_fraud"], test["ptype_prior_n"], global_rate)
    feature_cols += ["ptype_hist_rate"]

    # ---------- global expanding rate (train) / fixed final rate (test) ----------
    n_before = np.arange(len(train))
    fraud_before = train["is_fraud"].cumsum().values - train["is_fraud"].values
    train["global_hist_rate"] = _smoothed_rate(fraud_before, n_before, global_rate)
    test["global_hist_rate"] = global_rate  # by definition, all of train precedes all of test
    feature_cols += ["global_hist_rate"]

    # ---------- row-local features already built in data.py ----------
    feature_cols += [
        "tenure_days", "days_since_purchase", "claim_amount_inr", "claim_to_price_ratio",
        "customer_prior_claims", "photo_attached_bin", "partner_inspected_bin",
        "is_small_claim", "post_inspection_policy_change", "desc_len", "warranty_months",
    ]
    for c in ["is_small_claim", "post_inspection_policy_change"]:
        train[c] = train[c].astype(int)
        test[c] = test[c].astype(int)

    meta = {"global_rate": global_rate, "smoothing_alpha": SMOOTHING_ALPHA, "feature_cols": feature_cols}
    return train, test, feature_cols, meta


if __name__ == "__main__":
    import config
    from data import load_and_clean

    config.require_data_files()
    train_clean, test_clean, audit = load_and_clean(
        str(config.TRAIN_PATH), str(config.TEST_PATH),
        str(config.PARTNERS_PATH), str(config.PRODUCTS_PATH),
    )
    train_feat, test_feat, feature_cols, meta = build_features(train_clean, test_clean)
    print("global_rate: {:.4f}%".format(100 * meta["global_rate"]))
    print("feature columns:", feature_cols)
    print(train_feat[feature_cols].describe().T[["mean", "std", "min", "max"]])
    print("\ncold-start test partners (partner_id_prior_n == 0):",
          (test_feat["partner_id_prior_n"] == 0).sum())
    print("\nsample rows:")
    print(train_feat[["claim_id", "submitted_at", "partner_id", "is_fraud",
                       "partner_id_prior_n", "partner_id_prior_fraud", "partner_id_hist_rate"]].tail(10))
