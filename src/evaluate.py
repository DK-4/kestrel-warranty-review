"""
evaluate.py — Runs the temporal validation experiment end to end and
prints/saves an honest comparison of every model in train.MODEL_REGISTRY.

This is the ONLY place these numbers get computed. README.md and the
memo must quote numbers from reports/validation_report.json, generated
by running this script — never retyped by hand.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from data import load_and_clean
from features import build_features
from validate import temporal_split, VALIDATION_MONTHS
from train import fit_all
import metrics as M


def run():
    config.require_data_files()
    train_clean, test_clean, audit = load_and_clean(
        str(config.TRAIN_PATH), str(config.TEST_PATH),
        str(config.PARTNERS_PATH), str(config.PRODUCTS_PATH),
    )
    train_feat, test_feat, feature_cols, meta = build_features(train_clean, test_clean)

    fit_df, val_df = temporal_split(train_feat)
    print(f"fit period:   {fit_df['submitted_at'].min()} -> {fit_df['submitted_at'].max()}  "
          f"({len(fit_df)} claims, {int(fit_df['is_fraud'].sum())} fraud)")
    print(f"val period:   {val_df['submitted_at'].min()} -> {val_df['submitted_at'].max()}  "
          f"({len(val_df)} claims, {int(val_df['is_fraud'].sum())} fraud)")
    print(f"months in val period: {len(VALIDATION_MONTHS)}\n")

    models = fit_all(fit_df, feature_cols)

    X_val = val_df[feature_cols]
    y_val = val_df["is_fraud"].values
    amt_val = val_df["claim_amount_inr"].values

    results = {}
    for name, model in models.items():
        score = model.predict_proba_positive(X_val)
        report = M.full_report(y_val, score, amt_val, months_in_period=len(VALIDATION_MONTHS))
        results[name] = report

    print_comparison_table(results)

    # Champion selection: PR-AUC / precision@k / net value at k=120 all point
    # the same direction (both HGB variants > logistic regression > baseline).
    # Between the two HGB variants, ranking quality is effectively tied and
    # neither shows a calibration problem (checked directly: predicted-score
    # percentiles are well spread for both, not compressed into a narrow
    # band) -- so we keep the simpler, unweighted model rather than add the
    # class_weight parameter for no measured benefit.
    champion_name = "hist_gradient_boosting"
    champion = models[champion_name]
    champion_score = champion.predict_proba_positive(X_val)

    k_values_total = [10, 20, 30, 40, 50, 60, 80, 100, 120]  # total claims flagged across the WHOLE val window (3 months)
    sweep = M.k_sweep(y_val, champion_score, amt_val, k_values_total)
    n_months = len(VALIDATION_MONTHS)
    print(f"\n--- Net value vs. investigation capacity used ({champion_name}) ---")
    print("(k below = TOTAL claims flagged across the 3-month validation window, not per month)")
    print(f"{'k_total':>8}{'k/month':>9}{'fraud_caught':>14}{'gross_INR':>12}{'inv_cost':>10}{'goodwill':>10}{'net_INR':>12}")
    best = max(sweep, key=lambda r: r["net_value_inr"])
    for r in sweep:
        marker = "  <-- peak" if r["k"] == best["k"] else ""
        print(f"{r['k']:>8}{r['k']/n_months:>9.1f}{r['n_tp_in_topk']:>14}{r['gross_fraud_value_caught_inr']:>12,.0f}"
              f"{r['investigation_cost_inr']:>10,.0f}{r['goodwill_cost_inr']:>10,.0f}"
              f"{r['net_value_inr']:>12,.0f}{marker}")

    recommended_monthly_k = round(best["k"] / n_months)
    print(f"\nNet value peaks at k={best['k']} total over the 3-month window "
          f"(~{recommended_monthly_k}/month, vs. the full 40/month = 120-total capacity), "
          f"net INR {best['net_value_inr']:,.0f}.")
    print("This means: always filling all 40 monthly slots is NOT the profit-maximizing "
          "policy at the current claim-value profile -- a confidence threshold that only "
          "sends the highest-scored claims to investigation, even if that's fewer than "
          "capacity, outperforms always using full capacity.")

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.REPORTS_DIR / "validation_report.json"
    with open(out_path, "w") as f:
        json.dump({
            "validation_months": VALIDATION_MONTHS,
            "fit_period": {"start": str(fit_df["submitted_at"].min()), "end": str(fit_df["submitted_at"].max()),
                           "n_claims": len(fit_df), "n_fraud": int(fit_df["is_fraud"].sum())},
            "val_period": {"start": str(val_df["submitted_at"].min()), "end": str(val_df["submitted_at"].max()),
                            "n_claims": len(val_df), "n_fraud": int(val_df["is_fraud"].sum())},
            "feature_cols": feature_cols,
            "results": results,
            "champion_model": champion_name,
            "k_sweep": sweep,
            "k_sweep_peak": best,
            # predict.py reads this directly -- never hardcode this number
            # elsewhere. It's regenerated fresh every time evaluate.py runs,
            # so it always reflects the model actually fit on THIS machine,
            # not a number carried over from a different environment (HGB
            # results are not guaranteed bit-identical across sklearn
            # versions even with a fixed random_state -- confirmed in
            # practice: 1.8.0 vs 1.9.1 gave k=20 vs k=10 as the peak).
            "recommended_monthly_k": recommended_monthly_k,
        }, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")
    return results, champion_name, sweep


def print_comparison_table(results: dict):
    print(f"{'MODEL':<32}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>8}"
          f"{'ROC-AUC':>9}{'PR-AUC':>8}{'P@k':>8}{'FraudCaught':>13}{'NetValue(INR)':>16}")
    for name, r in results.items():
        prec = "n/a" if r["precision"] is None else f"{r['precision']:.3f}"
        f1 = "n/a" if r["f1"] is None else f"{r['f1']:.3f}"
        roc = "n/a" if r["roc_auc"] is None else f"{r['roc_auc']:.3f}"
        pr = "n/a" if r["pr_auc"] is None else f"{r['pr_auc']:.3f}"
        pk = r["precision_at_k"]["precision_at_k"]
        pk_str = "n/a" if pk is None else f"{pk:.3f}"
        fraud_caught = r["precision_at_k"]["fraud_caught"]
        fraud_total = r["precision_at_k"]["fraud_total"]
        net = r["rupee_impact"]["net_value_inr"]
        print(f"{name:<32}{r['accuracy']*100:>9.1f}%{prec:>11}{r['recall']:>9.3f}{f1:>8}"
              f"{roc:>9}{pr:>8}{pk_str:>8}{f'{fraud_caught}/{fraud_total}':>13}{net:>16,.0f}")


if __name__ == "__main__":
    run()
