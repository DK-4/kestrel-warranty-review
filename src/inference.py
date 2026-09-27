"""
inference.py — Score ONE new claim the same way predict.py scores the
whole test set, so the API and the UI can never drift from the batch
pipeline's behavior. Both api.py and streamlit_app.py import ClaimScorer
from here; neither re-implements any feature logic itself.

Point-in-time note for live scoring: a brand-new claim arriving today is,
by construction, always "in the future" relative to the training data.
So exactly like test.csv was scored in predict.py, a live claim's
partner/partner_type historical rates use the FINAL frozen statistics
computed from the training set as of the last training run — not
something recomputed per request. This is correct and leakage-safe, but
it does mean the model should be retrained periodically as new decided
claims accumulate; this API does not do that automatically.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

import config
from data import sanitize_claim_description, normalize_serial
from features import _final_group_stats, _smoothed_rate, SMOOTHING_ALPHA
from explain import explain_claim, risk_tier


class ClaimScorer:
    def __init__(self):
        model_path = config.MODELS_DIR / "model.joblib"
        meta_path = config.MODELS_DIR / "model_meta.json"
        if not model_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run `python src/predict.py` first "
                "to train and save the model this service depends on."
            )
        self.model = joblib.load(model_path)
        with open(meta_path) as f:
            self.meta = json.load(f)
        self.feature_cols = self.meta["feature_cols"]
        self.global_rate = self.meta["global_rate"]

        # Reference tables + frozen point-in-time lookup stats, built once
        # at startup from the same cleaned training set predict.py used.
        from data import load_and_clean
        train_clean, _, _ = load_and_clean(
            str(config.TRAIN_PATH), str(config.TEST_PATH),
            str(config.PARTNERS_PATH), str(config.PRODUCTS_PATH),
        )
        self.partners = pd.read_csv(config.PARTNERS_PATH)
        self.partners["onboarded_date"] = pd.to_datetime(self.partners["onboarded_date"])
        self.products = pd.read_csv(config.PRODUCTS_PATH)

        self.partner_stats = _final_group_stats(train_clean, "partner_id")
        self.ptype_stats = _final_group_stats(train_clean, "partner_type")

    def _partner_rate(self, partner_id: str) -> tuple[float, int]:
        if partner_id in self.partner_stats.index:
            row = self.partner_stats.loc[partner_id]
            n, fraud = float(row["n"]), float(row["fraud"])
        else:
            n, fraud = 0.0, 0.0  # cold-start partner, unseen in training
        rate = _smoothed_rate(fraud, n, self.global_rate, alpha=SMOOTHING_ALPHA)
        return rate, int(n)

    def _ptype_rate(self, partner_type: str) -> float:
        if partner_type in self.ptype_stats.index:
            row = self.ptype_stats.loc[partner_type]
            n, fraud = float(row["n"]), float(row["fraud"])
        else:
            n, fraud = 0.0, 0.0
        return _smoothed_rate(fraud, n, self.global_rate, alpha=SMOOTHING_ALPHA)

    def score(self, claim: dict) -> dict:
        """claim must have the same fields as a test_unlabelled.csv row
        (everything except is_fraud). Returns fraud_score, risk tier, and
        up to 3 rule-based reasons."""
        row = dict(claim)
        submitted_at = pd.to_datetime(row["submitted_at"])

        partner_match = self.partners[self.partners["partner_id"] == row["partner_id"]]
        if partner_match.empty:
            raise ValueError(f"Unknown partner_id: {row['partner_id']!r} — not in partners.csv")
        partner_row = partner_match.iloc[0]

        product_match = self.products[self.products["sku"] == row["sku"]]
        if product_match.empty:
            raise ValueError(f"Unknown sku: {row['sku']!r} — not in products.csv")
        product_row = product_match.iloc[0]

        desc_clean, _ = sanitize_claim_description(pd.Series([row.get("claim_description", "")]))
        desc_clean = desc_clean.iloc[0]

        tenure_days = (submitted_at - partner_row["onboarded_date"]).days
        is_small_claim = int(row["claim_amount_inr"] < 2000)
        post_change = int(submitted_at >= pd.Timestamp("2026-05-01"))
        photo_bin = int(str(row.get("photo_attached", "N")).upper() == "Y")
        inspected_bin = int(str(row.get("partner_inspected", "N")).upper() == "Y")
        claim_to_price = row["claim_amount_inr"] / product_row["list_price_inr"]

        partner_rate, partner_prior_n = self._partner_rate(row["partner_id"])
        ptype_rate = self._ptype_rate(partner_row["partner_type"])

        features = {
            "partner_id_hist_rate": partner_rate,
            "partner_id_prior_n": partner_prior_n,
            "ptype_hist_rate": ptype_rate,
            "global_hist_rate": self.global_rate,
            "tenure_days": tenure_days,
            "days_since_purchase": row["days_since_purchase"],
            "claim_amount_inr": row["claim_amount_inr"],
            "claim_to_price_ratio": claim_to_price,
            "customer_prior_claims": row.get("customer_prior_claims", 0),
            "photo_attached_bin": photo_bin,
            "partner_inspected_bin": inspected_bin,
            "is_small_claim": is_small_claim,
            "post_inspection_policy_change": post_change,
            "desc_len": len(desc_clean),
            "warranty_months": product_row["warranty_months"],
        }
        X = pd.DataFrame([features])[self.feature_cols]
        fraud_score = float(self.model.predict_proba_positive(X)[0])

        risk = risk_tier(fraud_score)

        explain_row = pd.Series({
            **features,
            "claim_amount_inr": row["claim_amount_inr"],
            "list_price_inr": product_row["list_price_inr"],
        })
        reasons = explain_claim(explain_row, self.global_rate, fraud_score=fraud_score, risk=risk, top_n=3)

        return {
            "claim_id": row.get("claim_id"),
            "fraud_score": round(fraud_score, 4),
            "risk": risk,
            "reasons": reasons,
        }
