"""
validate.py — Temporal train/validation split.

Validation window = the last 3 months of the train period (Apr-Jun 2026),
matching the length of the actual test period (Jul-Sep 2026) so the
validation experiment structurally mirrors the real train->test
transition, not an arbitrary split.

Because features.py already builds every group-level feature as a
strictly-prior expanding window over the FULL sorted train set, slicing
by date here introduces no extra leakage: a claim in the validation tail
only ever "sees" claims before it in time, which by construction fall in
the training portion. No feature needs to be recomputed for this split.
"""
import pandas as pd

VALIDATION_MONTHS = ["2026-04", "2026-05", "2026-06"]


def temporal_split(train_feat: pd.DataFrame):
    period = train_feat["submitted_at"].dt.to_period("M").astype(str)
    val_mask = period.isin(VALIDATION_MONTHS)
    fit = train_feat.loc[~val_mask].reset_index(drop=True)
    val = train_feat.loc[val_mask].reset_index(drop=True)
    assert fit["submitted_at"].max() < val["submitted_at"].min(), \
        "validation split is not strictly after the fit split"
    return fit, val
