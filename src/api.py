"""
api.py — POST /predict and GET /health.

Run with:  uvicorn src.api:app --reload --app-dir .
(or, from inside src/:  uvicorn api:app --reload)

The fraud score itself is a deterministic tabular ML model
(HistGradientBoostingClassifier) -- nothing here calls an LLM. Reasons
are rule-based (src/explain.py).
"""
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from inference import ClaimScorer

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

app = FastAPI(
    title="Kestrel Warranty Claim Fraud Scoring",
    description="Deterministic tabular ML fraud-risk score + rule-based reasons. No LLM in the scoring path.",
    version="1.0",
)

_scorer: Optional[ClaimScorer] = None


def get_scorer() -> ClaimScorer:
    global _scorer
    if _scorer is None:
        _scorer = ClaimScorer()
    return _scorer


class ClaimIn(BaseModel):
    claim_id: str
    submitted_at: str = Field(..., description="ISO timestamp, e.g. 2026-07-15T10:30:00")
    partner_id: str
    sku: str
    product_serial: str
    days_since_purchase: int
    claim_amount_inr: float
    photo_attached: str = Field(..., description="'Y' or 'N'")
    partner_inspected: str = Field(..., description="'Y' or 'N'")
    claim_description: str = ""
    inspector_note: str = ""
    customer_prior_claims: int = 0

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "claim_id": "WC999999",
            "submitted_at": "2026-07-15T10:30:00",
            "partner_id": "SP3207",
            "sku": "KH-AF-01",
            "product_serial": "KH123456789",
            "days_since_purchase": 45,
            "claim_amount_inr": 1500,
            "photo_attached": "N",
            "partner_inspected": "N",
            "claim_description": "power button not working",
            "inspector_note": "",
            "customer_prior_claims": 0,
        }
    })


class ClaimOut(BaseModel):
    claim_id: str
    fraud_score: float
    risk: str
    reasons: list[str]


@app.get("/health")
def health():
    try:
        get_scorer()
        return {"status": "ok", "model": "hist_gradient_boosting", "time": datetime.now(timezone.utc).isoformat()}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.post("/predict", response_model=ClaimOut)
def predict(claim: ClaimIn):
    scorer = get_scorer()
    try:
        result = scorer.score(claim.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return result
