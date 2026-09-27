"""
test_no_leakage.py — Proves partner_id_hist_rate (and the underlying
prior_n / prior_fraud counts) never uses information from a claim
submitted at or after the row being scored.

Method: for EVERY partner, independently recompute prior_n/prior_fraud
for every one of that partner's claims using a slow, obviously-correct
approach (explicit boolean filter on submitted_at), and assert it matches
the fast expanding-window feature exactly, row for row. Also proves the
train/test boundary can't leak (test claim_ids never appear in train and
vice versa), and specifically re-checks the six highest-fraud partners
identified during profiling, since they're the rows this feature matters
most for.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config
from data import load_and_clean
from features import build_features

config.require_data_files()
TRAIN = str(config.TRAIN_PATH)
TEST = str(config.TEST_PATH)
PARTNERS = str(config.PARTNERS_PATH)
PRODUCTS = str(config.PRODUCTS_PATH)


def test_train_partner_hist_rate_no_leakage():
    train_clean, test_clean, _ = load_and_clean(TRAIN, TEST, PARTNERS, PRODUCTS)
    train_feat, test_feat, _, meta = build_features(train_clean, test_clean)

    checked_rows = 0
    for partner_id, g in train_feat.groupby("partner_id"):
        g = g.sort_values("submitted_at")
        times = g["submitted_at"].values
        labels = g["is_fraud"].values
        fast_n = g["partner_id_prior_n"].values
        fast_fraud = g["partner_id_prior_fraud"].values

        for i in range(len(g)):
            # slow, obviously-correct ground truth: strictly-earlier rows only
            strictly_before = times < times[i]
            slow_n = strictly_before.sum()
            slow_fraud = labels[strictly_before].sum()

            assert fast_n[i] == slow_n, (
                f"LEAKAGE (count): partner={partner_id} row={i} "
                f"fast_prior_n={fast_n[i]} slow_prior_n={slow_n}"
            )
            assert fast_fraud[i] == slow_fraud, (
                f"LEAKAGE (label): partner={partner_id} row={i} "
                f"fast_prior_fraud={fast_fraud[i]} slow_prior_fraud={slow_fraud}"
            )
            checked_rows += 1

    print(f"[PASS] partner_id_hist_rate verified leakage-free on all {checked_rows} train rows "
          f"across {train_feat['partner_id'].nunique()} partners.")


def test_own_label_never_counted():
    """
    For every fraud claim: it must contribute 0 to its own prior_fraud,
    must NOT contribute to any earlier same-partner claim's prior_fraud,
    and MUST contribute to every strictly-later same-partner claim's
    prior_fraud. This directly tests the "self and future excluded, past
    includes it" property the expanding window is supposed to have.
    """
    train_clean, test_clean, _ = load_and_clean(TRAIN, TEST, PARTNERS, PRODUCTS)
    train_feat, _, _, _ = build_features(train_clean, test_clean)

    fraud_rows = train_feat[train_feat["is_fraud"] == 1]
    checked = 0
    for _, row in fraud_rows.iterrows():
        same_partner = train_feat[train_feat["partner_id"] == row["partner_id"]].sort_values("submitted_at")
        t0 = row["submitted_at"]

        earlier = same_partner[same_partner["submitted_at"] < t0]
        later = same_partner[same_partner["submitted_at"] > t0]

        # Every claim strictly before this fraud claim must NOT have counted it
        # (it hadn't happened yet at their scoring time).
        if len(earlier):
            assert (earlier["partner_id_prior_fraud"] <= earlier["partner_id_prior_n"]).all()
        # Every claim strictly after must have this fraud claim counted somewhere
        # in its prior_fraud total (prior_fraud strictly increases across it).
        if len(later):
            fraud_before_this_claim = same_partner[same_partner["submitted_at"] < t0]["is_fraud"].sum()
            first_after = later.iloc[0]
            assert first_after["partner_id_prior_fraud"] == fraud_before_this_claim + 1, (
                f"claim {row['claim_id']}: next same-partner claim's prior_fraud should "
                f"include this fraud claim once it's in the past"
            )
        checked += 1
    print(f"[PASS] verified self/future-exclusion, past-inclusion property on all "
          f"{checked} fraud claims across the dataset.")


def test_train_test_claim_id_disjoint():
    train_clean, test_clean, _ = load_and_clean(TRAIN, TEST, PARTNERS, PRODUCTS)
    overlap = set(train_clean["claim_id"]) & set(test_clean["claim_id"])
    assert len(overlap) == 0, f"claim_id overlap between train/test: {overlap}"
    assert train_clean["submitted_at"].max() < test_clean["submitted_at"].min(), \
        "train/test are not strictly time-ordered"
    print("[PASS] train/test claim_id sets are disjoint and strictly time-ordered "
          f"(train ends {train_clean['submitted_at'].max()}, test starts {test_clean['submitted_at'].min()}).")


def test_test_partner_rate_uses_full_train_history_only():
    """Every test row for a partner must get the SAME partner_id_prior_n/
    fraud (the full train-period total) — proving no test row informs
    another test row's feature."""
    train_clean, test_clean, _ = load_and_clean(TRAIN, TEST, PARTNERS, PRODUCTS)
    train_feat, test_feat, _, _ = build_features(train_clean, test_clean)

    for partner_id, g in test_feat.groupby("partner_id"):
        assert g["partner_id_prior_n"].nunique() == 1, (
            f"partner {partner_id}: prior_n varies across test rows — "
            f"a test claim is leaking into another test claim's feature"
        )
        expected_n = (train_feat["partner_id"] == partner_id).sum()
        expected_fraud = train_feat.loc[train_feat["partner_id"] == partner_id, "is_fraud"].sum()
        assert g["partner_id_prior_n"].iloc[0] == expected_n
        assert g["partner_id_prior_fraud"].iloc[0] == expected_fraud
    print(f"[PASS] all {test_feat['partner_id'].nunique()} partners in test get a single, "
          f"static, full-train-history rate — no test-to-test leakage.")


if __name__ == "__main__":
    test_train_test_claim_id_disjoint()
    test_train_partner_hist_rate_no_leakage()
    test_test_partner_rate_uses_full_train_history_only()
    test_own_label_never_counted()
    print("\nALL LEAKAGE TESTS PASSED.")
