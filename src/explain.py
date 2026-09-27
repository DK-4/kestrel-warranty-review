"""
explain.py — Deterministic, rule-based reasons for a claim's fraud score.

No LLM anywhere in this file. Every reason is a plain-language rendering
of an actual feature value on the actual row, compared to a fixed,
documented reference point (the global rate, a policy threshold, etc.).
Reasons are ranked by a fixed priority order below (highest-signal
features first, per the validation findings), not by re-deriving
importance per row -- keeps output deterministic and auditable.
"""
from __future__ import annotations

import pandas as pd


def _reasons_for_row(row: pd.Series, global_rate: float) -> list[tuple[float, str]]:
    """Returns (priority, text) tuples; higher priority = shown first."""
    reasons = []

    # 1. Partner history -- the single strongest signal found in validation.
    prior_n = row["partner_id_prior_n"]
    hist_rate = row["partner_id_hist_rate"]
    if prior_n == 0:
        reasons.append((
            9.0,
            "This partner has no prior claim history in our records (newly onboarded "
            "or first claim) -- risk is estimated from partner type and overall averages only."
        ))
    elif hist_rate > 3 * global_rate and prior_n >= 5:
        reasons.append((
            10.0,
            f"This partner's historical fraud rate is {hist_rate*100:.1f}% over "
            f"{int(prior_n)} prior claims, well above the {global_rate*100:.1f}% company average."
        ))
    elif hist_rate > 1.5 * global_rate and prior_n >= 5:
        reasons.append((
            6.0,
            f"This partner's historical fraud rate ({hist_rate*100:.1f}% over {int(prior_n)} "
            f"prior claims) is somewhat above the {global_rate*100:.1f}% company average."
        ))

    # 2. Claim amount vs. product list price.
    ratio = row["claim_to_price_ratio"]
    if pd.notna(ratio) and ratio >= 0.8:
        reasons.append((
            7.0,
            f"Claim amount (Rs {row['claim_amount_inr']:,.0f}) is {ratio*100:.0f}% of this "
            f"product's list price (Rs {row['list_price_inr']:,.0f}) -- unusually high relative to the item."
        ))

    # 3. Small claim, uninspected, post auto-approval policy change.
    if row["is_small_claim"] and row["post_inspection_policy_change"] and row["partner_inspected_bin"] == 0:
        reasons.append((
            5.0,
            "Small claim (under Rs 2,000) submitted after the 1 May 2026 policy change, "
            "so it was auto-approved without inspection sign-off."
        ))

    # 4. Partner tenure.
    tenure_days = row["tenure_days"]
    if pd.notna(tenure_days) and 90 <= tenure_days <= 180:
        reasons.append((
            4.0,
            f"Partner was onboarded {int(tenure_days)} days ago -- validation found the "
            f"3-6 month window for newly onboarded authorised service centres carries "
            f"elevated risk (this is a narrow, specific pattern, not true of new partners generally)."
        ))

    # 5. No photo attached.
    if row["photo_attached_bin"] == 0:
        reasons.append((3.0, "No supporting photo was attached to this claim."))

    # 6. Customer with many prior claims.
    if pd.notna(row["customer_prior_claims"]) and row["customer_prior_claims"] >= 3:
        reasons.append((
            3.5,
            f"Customer has {int(row['customer_prior_claims'])} prior warranty claims on record."
        ))

    if not reasons:
        reasons.append((0.0, "No individual risk factor stands out; score reflects a "
                              "combination of smaller signals below the reporting threshold."))

    return sorted(reasons, key=lambda x: -x[0])


def explain_claim(row: pd.Series, global_rate: float, fraud_score: float = None,
                   risk: str = None, top_n: int = 3) -> list[str]:
    reasons = [text for _, text in _reasons_for_row(row, global_rate)[:top_n]]
    # Individual rule-based flags can fire (e.g. a bad partner history) even
    # when the model's overall score is low -- the model weighs many
    # features together and can be pulled down by others (claim size,
    # inspection regime, etc.). Without this, an employee could see a
    # "high partner fraud rate" bullet on a claim scored 0.3% and
    # reasonably distrust the tool. State the overall verdict first so the
    # bullets read as context, not as the verdict itself.
    flagged_language = any(w in " ".join(reasons) for w in ["elevated", "above the", "well above"])
    if risk == "low" and fraud_score is not None and flagged_language:
        headline = (
            f"Overall model score is LOW ({fraud_score*100:.1f}%) despite some individual "
            f"factors below -- no single factor was decisive; the combined assessment across "
            f"all features placed this in the low-risk range."
        )
        reasons = [headline] + reasons
        reasons = reasons[:max(top_n, 2)]
    return reasons


def risk_tier(fraud_score: float) -> str:
    """Single source of truth for risk-tier thresholds -- used by both
    inference.py (live scoring) and explain_dataframe (batch scoring)."""
    if fraud_score >= 0.2:
        return "high"
    elif fraud_score >= 0.05:
        return "medium"
    return "low"


def explain_dataframe(df: pd.DataFrame, global_rate: float, score_col: str = None, top_n: int = 3) -> pd.Series:
    if score_col is not None:
        return df.apply(
            lambda row: " | ".join(explain_claim(
                row, global_rate, fraud_score=row[score_col],
                risk=risk_tier(row[score_col]), top_n=top_n
            )), axis=1
        )
    return df.apply(lambda row: " | ".join(explain_claim(row, global_rate, top_n=top_n)), axis=1)
