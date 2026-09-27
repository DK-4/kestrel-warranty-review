"""
metrics.py — Single source of truth for every evaluation number.

Every number that ends up in the README, the memo, or reports/*.json
must come from calling these functions on the validation set, not be
retyped by hand anywhere. That's the discipline this project got
dinged for skipping last time.

Two DIFFERENT rupee numbers are reported, deliberately kept separate
rather than netted into one, because they answer two different
questions:

  gross_fraud_value_caught  — "how much fraud did we stop, in rupees,
                               among the claims we checked" (Farhan's
                               literal question)
  net_value                 — gross_fraud_value_caught minus the
                               investigation cost of every claim checked
                               and the goodwill cost of every legitimate
                               claim checked (the fuller business-impact
                               number)

Both are defined ONCE, here, and every caller must use these functions
rather than recomputing the formula inline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)

GOODWILL_COST_INR = 380       # ops-policy §4: cost of holding a genuine claim for review
INVESTIGATION_COST_INR = 260  # ops-policy §4: blended cost of a service contact
MONTHLY_CAPACITY = 40         # ops-policy §5: investigation desk capacity


def standard_metrics(y_true, y_score, threshold=0.5) -> dict:
    """Accuracy/precision/recall/F1 at a fixed threshold, plus threshold-free
    ranking metrics (ROC-AUC, PR-AUC). Handles the undefined case (a
    constant-score model, e.g. the do-nothing baseline) explicitly rather
    than silently reporting a misleading 0."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    n_predicted_positive = y_pred.sum()
    constant_score = np.unique(y_score).size == 1

    out = {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "n_predicted_positive": int(n_predicted_positive),
    }

    if n_predicted_positive == 0:
        out["precision"] = None  # undefined: 0/0, not 0 — no positives were predicted at all
        out["recall"] = 0.0      # well-defined: 0 of the actual positives were caught
        out["f1"] = None
    else:
        out["precision"] = precision_score(y_true, y_pred, zero_division=0)
        out["recall"] = recall_score(y_true, y_pred, zero_division=0)
        out["f1"] = f1_score(y_true, y_pred, zero_division=0)

    if constant_score or y_true.sum() == 0 or y_true.sum() == len(y_true):
        out["roc_auc"] = None   # undefined: model can't rank if every score is identical
        out["pr_auc"] = None
    else:
        out["roc_auc"] = roc_auc_score(y_true, y_score)
        out["pr_auc"] = average_precision_score(y_true, y_score)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out["confusion_matrix"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    return out


def precision_at_k(y_true, y_score, k: int) -> dict:
    """Among the top-k highest-scored claims, what fraction are actually fraud."""
    order = np.argsort(-np.asarray(y_score))
    top_k_idx = order[:k]
    y_true = np.asarray(y_true)
    top_k_true = y_true[top_k_idx]
    n_fraud_total = int(y_true.sum())
    n_fraud_caught = int(top_k_true.sum())
    return {
        "k": k,
        "precision_at_k": n_fraud_caught / k if k > 0 else None,
        "fraud_caught": n_fraud_caught,
        "fraud_total": n_fraud_total,
        "recall_at_k": n_fraud_caught / n_fraud_total if n_fraud_total > 0 else None,
        "top_k_idx": top_k_idx,  # positions into the arrays passed in, for the rupee calc
    }


def rupee_impact(y_true, y_score, claim_amount, k: int) -> dict:
    """
    gross_fraud_value_caught = sum(claim_amount) over true-positive claims in the top-k
    net_value = gross_fraud_value_caught
                - (k * INVESTIGATION_COST_INR)                  # cost of checking all k
                - (n_false_positives_in_top_k * GOODWILL_COST_INR)  # goodwill hit on legit ones held
    """
    y_true = np.asarray(y_true)
    claim_amount = np.asarray(claim_amount, dtype=float)
    pak = precision_at_k(y_true, y_score, k)
    idx = pak["top_k_idx"]

    is_fraud_in_topk = y_true[idx] == 1
    n_fp_in_topk = int((~is_fraud_in_topk).sum())
    n_tp_in_topk = int(is_fraud_in_topk.sum())

    gross_fraud_value_caught = float(claim_amount[idx][is_fraud_in_topk].sum())
    investigation_cost = k * INVESTIGATION_COST_INR
    goodwill_cost = n_fp_in_topk * GOODWILL_COST_INR
    net_value = gross_fraud_value_caught - investigation_cost - goodwill_cost

    # For context: total fraud value that existed in the period, and how much
    # of it slipped through (fraud claims NOT in the top-k -> still get paid).
    all_fraud_idx = np.where(y_true == 1)[0]
    total_fraud_value = float(claim_amount[all_fraud_idx].sum())
    caught_idx = set(idx[is_fraud_in_topk].tolist())
    missed_fraud_value = float(sum(
        claim_amount[i] for i in all_fraud_idx if i not in caught_idx
    ))

    return {
        "k": k,
        "n_tp_in_topk": n_tp_in_topk,
        "n_fp_in_topk": n_fp_in_topk,
        "gross_fraud_value_caught_inr": gross_fraud_value_caught,
        "investigation_cost_inr": investigation_cost,
        "goodwill_cost_inr": goodwill_cost,
        "net_value_inr": net_value,
        "total_fraud_value_in_period_inr": total_fraud_value,
        "missed_fraud_value_inr": missed_fraud_value,
    }


def full_report(y_true, y_score, claim_amount, months_in_period: float, threshold=0.5) -> dict:
    """Runs the whole metric suite for one model on one evaluation period.
    k scales with the length of the period (MONTHLY_CAPACITY * months)."""
    k = round(MONTHLY_CAPACITY * months_in_period)
    report = standard_metrics(y_true, y_score, threshold=threshold)
    report["precision_at_k"] = precision_at_k(y_true, y_score, k)
    report["precision_at_k"].pop("top_k_idx")  # not JSON-serializable / not needed in the report
    report["rupee_impact"] = rupee_impact(y_true, y_score, claim_amount, k)
    return report


def k_sweep(y_true, y_score, claim_amount, k_values) -> list[dict]:
    """Net value at a range of operating points, so we can check whether
    the assumption 'always use the full monthly capacity' is actually the
    profit-maximizing choice, rather than taking it for granted."""
    return [rupee_impact(y_true, y_score, claim_amount, k) for k in k_values]
