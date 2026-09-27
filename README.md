# Kestrel Home — Warranty Claim Fraud Review

A small, leakage-aware fraud-risk ranking system for Kestrel Home warranty claims.

The system scores warranty claims by fraud risk, ranks them for investigation, provides human-readable reasons, generates the required `predictions.csv`, and exposes the model through a FastAPI endpoint with a simple Streamlit review screen.

---

## 1. Problem

Kestrel Home has approximately 15 months of historical warranty claims with investigation outcomes and wants to identify potentially fraudulent claims before payment.

The investigation team has limited review capacity, so the practical problem is not simply:

> "Classify every claim correctly."

It is:

> "Rank claims so investigators can review the highest-risk claims first."

The dataset contains a small fraud class. After cleaning and removing undecided claims, the training set contains:

- **11,146 decided claims**
- **141 fraud claims**
- **1.265% fraud rate**

Because fraud is rare, raw accuracy is not sufficient to judge usefulness. A majority-class model can already achieve roughly 98% accuracy while catching no fraud.

The project therefore evaluates both conventional classification metrics and ranking/business metrics.

---

## 2. What the system does

```text
Raw Kestrel claims
        |
        v
Data cleaning & validation
        |
        v
Deduplication / undecided-claim handling
        |
        v
Prompt-injection sanitization
        |
        v
Point-in-time feature engineering
        |
        v
Temporal validation
        |
        v
Fraud-risk model
        |
        +------------------+
        |                  |
        v                  v
 fraud score          human-readable reasons
        |
        v
rank claims for investigation
        |
        +----------------------+
        |                      |
        v                      v
predictions.csv        FastAPI / Streamlit
3. Main engineering decisions
Rare-event evaluation

Accuracy is reported because it is part of the client requirement, but it is not treated as the primary success metric.

The project also measures:

Precision
Recall
F1
ROC-AUC
PR-AUC
Precision at investigation capacity
Fraud value captured
Investigation cost
Goodwill cost
Net value
Temporal validation

The training data spans:

April 2025 → June 2026

The hidden test period is later:

July 2026 → September 2026

Therefore the primary validation is time-based rather than a random train/test split.

Current validation run:

Fit period:
2025-04-01 → 2026-03-31
9,015 claims
96 fraud

Validation period:
2026-04-01 → 2026-06-30
2,131 claims
45 fraud
Leakage-safe partner history

Partner historical fraud features are calculated using only earlier claims and exclude the current claim's own outcome.

The repository includes explicit leakage tests covering:

historical partner-rate construction
self-label exclusion
train/test claim-ID separation
test partner-rate construction from training history

Current test result:

4 passed
Untrusted claim text

Five input rows contained adversarial/prompt-injection style instructions inside claim_description.

These instructions are treated as untrusted data rather than model instructions, and the affected text is sanitized before downstream processing.

Partner-age hypothesis

The client suspected that newer partners were generally responsible for fraud.

The analysis did not assume this hypothesis. Partner tenure was tested against fraud behavior and the overall relationship was not strong enough to justify a blanket "new partner = fraud" rule.

A more specific elevated-risk pattern was observed among newly onboarded authorised service centres in the 3–6 month tenure window.

4. Data cleaning

The cleaning pipeline handles:

duplicate claims
undecided claims
serial normalization
timestamp parsing
partner/product joins
claim-description sanitization
validation assertions

Current cleaning output:

train_clean: 11,146 rows
test_clean:   2,252 rows

training duplicates removed: 681
unique undecided claims removed: 202
fraud rate: 1.265%

Undecided claims are not treated as non-fraud. They are excluded from supervised training/evaluation because there is no ground-truth outcome.

5. Features

Current model features include:

partner_id_hist_rate
partner_id_prior_n
ptype_hist_rate
global_hist_rate
tenure_days
days_since_purchase
claim_amount_inr
claim_to_price_ratio
customer_prior_claims
photo_attached_bin
partner_inspected_bin
is_small_claim
post_inspection_policy_change
desc_len
warranty_months

Historical partner and partner-type fraud signals are constructed point-in-time to reduce future-label leakage.

6. Models evaluated

The validation pipeline compares:

Do-nothing baseline
Logistic Regression
HistGradientBoostingClassifier
Class-weighted HistGradientBoostingClassifier

The current prediction pipeline uses:

HistGradientBoostingClassifier

as the production candidate.

7. Current validation results

Three-month temporal validation:

Model	Accuracy	Precision	Recall	F1	ROC-AUC	PR-AUC
Do nothing	97.9%	—	0.0%	—	—	—
Logistic Regression	97.5%	16.7%	4.4%	7.0%	0.700	0.101
HistGradientBoosting	97.8%	40.0%	8.9%	14.5%	0.659	0.110
HGB balanced	97.7%	38.5%	11.1%	17.2%	0.605	0.104

The validation results reinforce why accuracy alone is not sufficient for this problem.

8. Investigation-capacity analysis

Kestrel's investigation desk has a maximum monthly capacity of approximately 40 claims.

The current economic sensitivity analysis evaluates the highest-scored claims across the three-month validation window and reports the equivalent monthly capacity.

For the current HGB run:

20 total claims across the 3-month validation window
≈ 6.7 claims/month equivalent

Fraud caught: 7 / 45
Gross fraud value captured: ₹25,015
Net value after investigation + goodwill costs: ₹14,875

This is a pooled three-month sensitivity analysis rather than a guarantee of a fixed optimal monthly quota.

The current analysis suggests that aggressively filling all 40 investigation slots is not automatically value-maximizing under the observed claim-value/cost profile.

9. Prediction output

The final prediction pipeline retrains the selected model on the full decided training set:

11,146 claims
April 2025 → June 2026

It then scores the hidden test set:

2,252 claims
July 2026 → September 2026

The generated file is:

predictions.csv

with:

claim_id,score

where higher score means higher estimated fraud risk.

The current prediction file was validated 1:1 against sample_submission.csv.

10. Explainability

For selected high-risk claims, the system generates human-readable reasons based on actual model features.

Examples of evidence categories include:

elevated historical partner fraud rate
unusual claim-to-product-price ratio
partner tenure
small-claim / inspection-policy behavior
previous customer claim history

The explanation layer is deterministic and grounded in structured features.

An LLM is not required for the fraud score.

11. API

The project includes a FastAPI service.

Start API

From the project root:

uvicorn src.api:app --reload --app-dir .

Expected local address:

http://127.0.0.1:8000

The latest development run successfully started the FastAPI application.

The service exposes the model for single-record prediction.

12. Streamlit UI

A simple Streamlit screen is included for a Kestrel employee to submit a claim and view its fraud score and explanations.

Start UI
streamlit run src/streamlit_app.py

Expected local address:

http://localhost:8501

The latest development run successfully started the Streamlit application.

13. Installation

Create a virtual environment:

python -m venv .venv

Activate it on Windows:

.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

The current environment includes:

pandas
numpy
scikit-learn
joblib
pytest
FastAPI
Uvicorn
Streamlit
python-dotenv
14. Reproduce the pipeline
Step 1 — Clean the data
python src/data.py
Step 2 — Generate features
python src/features.py
Step 3 — Run leakage tests
pytest -v

Expected:

4 passed
Step 4 — Run temporal evaluation
python src/evaluate.py

This generates:

reports/validation_report.json
Step 5 — Train the final model and generate predictions
python src/predict.py

This generates:

predictions.csv
models/
reports/flagged_claims_with_reasons.csv
15. Project structure
kestrel/
│
├── data/                         # Client data - not committed
│
├── models/                       # Generated model artifacts
│
├── reports/
│   ├── validation_report.json
│   └── flagged_claims_with_reasons.csv
│
├── src/
│   ├── config.py
│   ├── data.py
│   ├── features.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── api.py
│   └── streamlit_app.py
│
├── tests/
│   └── test_no_leakage.py
│
├── predictions.csv
├── README.md
├── requirements.txt
├── .gitignore
└── .env.example
16. Known limitations

This is a small assessment prototype rather than a production fraud-decision system.

Known limitations include:

historical legacy_zoho labels may contain unresolved cases recorded as non-fraud because the legacy system could not store blanks
investigation-resolution timestamps are not available, so historical-outcome timing is an explicit assumption in the point-in-time feature design
some test partners are cold-start partners with no historical claims
model performance varies by validation period
small claims can be difficult for the model to rank correctly
the investigation-capacity analysis is a pooled three-month sensitivity analysis rather than a guarantee of a fixed monthly optimum
the model is intended to support human investigation, not automatically reject or deny warranty claims
17. Security and privacy

Kestrel data is client data.

The raw task data is intentionally excluded from Git tracking.

.gitignore excludes:

data/
.env

API keys or other secrets must not be committed.

The GitHub repository should remain private when client data is involved.

18. Design choices
Why no LLM for fraud scoring?

The core problem is structured tabular fraud ranking.

A local ML model provides:

reproducible scores
low inference cost
no paid API dependency
simpler deployment
easier evaluation

An LLM would add unnecessary cost and another failure mode without being required for the core prediction task.

Why not optimize only for 97% accuracy?

Because fraud is rare.

A model that predicts every claim as legitimate can achieve high accuracy while catching no fraud.

The system therefore focuses on ranking quality, fraud capture, and economic impact in addition to accuracy.

Why investigate partner age but not assume it?

The client explicitly suggested that newer partners may be responsible.

The data was tested rather than encoded into a fixed business rule.

19. Outputs

The project produces:

predictions.csv
reports/validation_report.json
reports/flagged_claims_with_reasons.csv
models/model.joblib
models/model_meta.json

These artifacts are generated by the pipeline and should be kept synchronized with the code version used for submission.

20. Assessment submission

The required submission artifacts are:

predictions.csv
working API
working Streamlit screen
evaluation evidence
one-page business memo
completed submission form
short screen recording

Repository:

https://github.com/DK-4/kestrel-warranty-review
