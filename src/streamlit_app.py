"""
streamlit_app.py — One page. Employee enters a claim, sees the fraud
score, risk tier, and up to 3 rule-based reasons. Uses the exact same
ClaimScorer as api.py -- no separate scoring logic to drift out of sync.

Run with:  streamlit run src/streamlit_app.py
"""
import sys
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
from inference import ClaimScorer

st.set_page_config(page_title="Kestrel Warranty Claim Review", layout="centered")


@st.cache_resource
def get_scorer():
    return ClaimScorer()


st.title("Kestrel Warranty Claim Review")
st.caption("Fraud-risk score from a deterministic tabular ML model. No LLM in the scoring path.")

try:
    scorer = get_scorer()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

with st.form("claim_form"):
    col1, col2 = st.columns(2)
    with col1:
        claim_id = st.text_input("Claim ID", value="WC999999")
        partner_id = st.text_input("Partner ID", value="SP3207")
        sku = st.text_input("SKU", value="KH-AF-01")
        product_serial = st.text_input("Product serial", value="KH123456789")
        submitted_date = st.date_input("Submitted date", value=date(2026, 7, 15))
    with col2:
        claim_amount_inr = st.number_input("Claim amount (Rs)", min_value=0.0, value=1500.0, step=100.0)
        days_since_purchase = st.number_input("Days since purchase", min_value=0, value=45)
        customer_prior_claims = st.number_input("Customer's prior claims", min_value=0, value=0)
        photo_attached = st.selectbox("Photo attached?", ["Y", "N"], index=1)
        partner_inspected = st.selectbox("Partner inspected?", ["Y", "N"], index=1)

    claim_description = st.selectbox(
        "Fault description",
        ["power button not working", "remote not working", "loud noise while running",
         "filter indicator stuck", "motor not running", "blade jammed", "water leaking",
         "burning smell", "not charging", "unit not heating", "tripping mcb"],
    )
    submitted = st.form_submit_button("Analyze claim")

if submitted:
    claim = {
        "claim_id": claim_id,
        "submitted_at": submitted_date.isoformat(),
        "partner_id": partner_id,
        "sku": sku,
        "product_serial": product_serial,
        "days_since_purchase": days_since_purchase,
        "claim_amount_inr": claim_amount_inr,
        "photo_attached": photo_attached,
        "partner_inspected": partner_inspected,
        "claim_description": claim_description,
        "inspector_note": "",
        "customer_prior_claims": customer_prior_claims,
    }
    try:
        result = scorer.score(claim)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    st.divider()
    risk_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}[result["risk"]]
    st.metric("Fraud score", f"{result['fraud_score']*100:.1f}%")
    st.write(f"**Risk: {risk_color} {result['risk'].upper()}**")
    st.write("**Why:**")
    for r in result["reasons"]:
        st.write(f"- {r}")
