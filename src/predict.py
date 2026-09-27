"""
predict.py — Retrain the champion model (HistGradientBoostingClassifier,
chosen in evaluate.py) on the FULL decided training set (all 15 months,
Apr 2025-Jun 2026 -- not just the fit-split used for validation), score
the real test set (Jul-Sep 2026), and write predictions.csv.

The validation numbers from evaluate.py are the honest estimate of how
this retrained model will perform -- test has no labels, so we cannot
compute precision@k or rupee value on it directly. What we CAN and do
report here: the score distribution, the score cutoff implied by the
validated-optimal operating point (k~20/month, not the full 40/month
capacity -- see reports/validation_report.json), and which/how many test
claims would be flagged for investigation at that recommended cutoff
each month.
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from data import load_and_clean
from features import build_features
from train import HGBModel
from explain import explain_dataframe

CHAMPION_MODEL_NAME = "hist_gradient_boosting"


def _load_recommended_k() -> int:
    """Read the recommended monthly operating point from evaluate.py's most
    recent output. Never hardcode this number here -- HGB results are not
    guaranteed bit-identical across scikit-learn versions even with a fixed
    random_state (confirmed in practice), so the right k can differ by
    machine. Always defer to whatever THIS machine's evaluate.py found."""
    report_path = config.REPORTS_DIR / "validation_report.json"
    if not report_path.exists():
        raise FileNotFoundError(
            f"{report_path} not found. Run `python src/evaluate.py` first -- "
            "predict.py needs its validated recommended_monthly_k, and that "
            "number depends on the model fit on THIS machine, so it can't "
            "be shipped as a constant."
        )
    with open(report_path) as f:
        report = json.load(f)
    if "recommended_monthly_k" not in report:
        raise KeyError(
            f"{report_path} is from an older version of evaluate.py "
            "without recommended_monthly_k. Re-run `python src/evaluate.py`."
        )
    return report["recommended_monthly_k"]


def run():
    config.require_data_files()
    recommended_monthly_k = _load_recommended_k()

    train_clean, test_clean, audit = load_and_clean(
        str(config.TRAIN_PATH), str(config.TEST_PATH),
        str(config.PARTNERS_PATH), str(config.PRODUCTS_PATH),
    )
    train_feat, test_feat, feature_cols, meta = build_features(train_clean, test_clean)
    global_rate = meta["global_rate"]

    print(f"Retraining {CHAMPION_MODEL_NAME} on the FULL decided training set: "
          f"{len(train_feat)} claims, {train_feat['submitted_at'].min()} -> "
          f"{train_feat['submitted_at'].max()}, fraud rate {global_rate*100:.3f}%")

    final_model = HGBModel().fit(train_feat[feature_cols], train_feat["is_fraud"].values)

    test_scores = final_model.predict_proba_positive(test_feat[feature_cols])
    test_feat = test_feat.copy()
    test_feat["fraud_score"] = test_scores

    # ---- predictions.csv, matching sample_submission.csv exactly ----
    sample_sub = pd.read_csv(config.SAMPLE_SUBMISSION_PATH)
    predictions = test_feat[["claim_id", "fraud_score"]].rename(columns={"fraud_score": "score"})
    predictions = sample_sub[["claim_id"]].merge(predictions, on="claim_id", how="left")

    assert predictions["claim_id"].tolist() == sample_sub["claim_id"].tolist(), \
        "predictions.csv row order does not match sample_submission.csv"
    assert predictions["score"].notna().all(), "some test claims got no prediction"
    assert len(predictions) == len(sample_sub), "row count mismatch vs sample_submission.csv"

    pred_path = config.PROJECT_ROOT / "predictions.csv"
    predictions.to_csv(pred_path, index=False)
    print(f"\nWrote {pred_path} ({len(predictions)} rows, validated 1:1 against sample_submission.csv)")

    # ---- test-set score summary (no ground truth available -- descriptive only) ----
    print("\nTest set fraud_score distribution:")
    print(test_feat["fraud_score"].describe())

    test_feat["month"] = test_feat["submitted_at"].dt.to_period("M")
    print(f"\nAt the VALIDATED-OPTIMAL operating point (top {recommended_monthly_k}/month by score, "
          f"not the raw 40/month capacity -- see reports/validation_report.json k-sweep):")
    for month, g in test_feat.groupby("month"):
        cutoff_row = g.nlargest(recommended_monthly_k, "fraud_score")
        cutoff_score = cutoff_row["fraud_score"].min()
        print(f"  {month}: {len(g)} claims -> flag top {recommended_monthly_k} "
              f"(score >= {cutoff_score:.4f}) for investigation")

    # ---- explanations for the flagged claims (rule-based, no LLM) ----
    flagged = pd.concat([
        g.nlargest(recommended_monthly_k, "fraud_score")
        for _, g in test_feat.groupby("month")
    ])
    flagged = flagged.copy()
    flagged["reasons"] = explain_dataframe(flagged, global_rate, score_col="fraud_score", top_n=3)

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    flagged_out = config.REPORTS_DIR / "flagged_claims_with_reasons.csv"
    flagged[["claim_id", "submitted_at", "partner_id", "claim_amount_inr",
             "fraud_score", "reasons"]].to_csv(flagged_out, index=False)
    print(f"\nWrote {flagged_out} ({len(flagged)} flagged claims with explanations)")

    # ---- save model + metadata ----
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, config.MODELS_DIR / "model.joblib")
    with open(config.MODELS_DIR / "model_meta.json", "w") as f:
        json.dump({
            "model": CHAMPION_MODEL_NAME,
            "trained_on_rows": len(train_feat),
            "trained_on_range": [str(train_feat["submitted_at"].min()), str(train_feat["submitted_at"].max())],
            "feature_cols": feature_cols,
            "global_rate": global_rate,
            "recommended_monthly_k": recommended_monthly_k,
        }, f, indent=2, default=str)
    print(f"Saved model + metadata to {config.MODELS_DIR}/")

    return predictions, flagged


if __name__ == "__main__":
    run()
