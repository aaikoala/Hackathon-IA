# Employee Attrition Prediction
### Binary Classification · Decision Support · HR Analytics
**Responsible AI — Cybersecurity · Ethics · Frugality · Explainability**

---

## 1. MODEL OBJECTIVE

* **Model name:** Employee Attrition Prediction v2.0
* **Use case / Task:** Binary classification — predicting employee resignation risk from structured HR data.
* **Inputs:** Salary, satisfaction score, engagement survey, absences, late days, tenure, department, performance score, special projects count, recruitment source, married status, age.
* **Outputs:** 1.  Resignation probability score [0–1]
    2.  Predicted class: Stay / Leave
    3.  Top 3 SHAP drivers in plain English
    4.  Predicted departure reason category (Voluntary / Involuntary / Retirement / Health)
* **Target users:** HR managers, CHROs — decision support only.
* **WARNING:** NOT for automated hiring, firing or pay decisions.

---

## 2. TRAINING DATA

* **Name / Source:** Human Resources Data Set — Kaggle (Drs. Rich Huebner & Carla Patalano)
* **Total samples:** 311 employees (after PII removal)
* **Class split:** 207 stayed (66.6%) / 104 left (33.4%) — moderate imbalance handled with `class_weight='balanced'`.
* **Diversity:** Single company, US-based, mixed departments (Sales, IT, Production, Admin, Executive). Age range: 26–65, mixed seniority levels.

**Features REMOVED before modelling:**
* `Employee_Name`, `SSN`, `DOB`, `ManagerName` (Privacy/PII)
* `RaceDesc`, `HispanicLatino` (Legal/Ethics)
* `EmploymentStatus`, `EmpStatusID` (Target Leakage)
* `MaritalDesc`, `MaritalStatusID` (Redundancy)
* `PerformanceScore` (text)

**Known limits:**
* Single company — may not generalise across industries or geographies.
* Synthetic dataset — not collected from a real organisation.
* No temporal dimension (no date of survey, etc.).

---

## 3. PERFORMANCE 
*(20% stratified test set — 63 employees)*

### Model Benchmark
* **Random Forest (Selected Model)**
    * **ROC-AUC:** ~0.88 | **F1-Score (Left):** ~0.80
    * **Footprint:** 0.000001 kg CO₂
    * **Status:** **Recommended** — Best balance of accuracy and stability.
* **Logistic Regression**
    * **ROC-AUC:** ~0.82 | **F1-Score (Left):** ~0.74
    * **Footprint:** 0.000001 kg CO₂
    * **Status:** **Best Robustness** — Highly reliable for resource-constrained environments.
* **Gradient Boosting**
    * **ROC-AUC:** ~0.86 | **F1-Score (Left):** ~0.78
    * **Footprint:** 0.000002 kg CO₂
    * **Status:** **High Accuracy** — Slightly higher compute footprint.
* **SVM (Support Vector Machine)**
    * **ROC-AUC:** ~0.83 | **F1-Score (Left):** ~0.75
    * **Footprint:** 0.000001 kg CO₂
    * **Status:** **Warning** — Sensitive to feature scaling and outliers.
* **Decision Tree**
    * **ROC-AUC:** ~0.75 | **F1-Score (Left):** ~0.69
    * **Footprint:** 0.000001 kg CO₂
    * **Status:** **Explainable** — Prone to overfitting on small datasets.

---

## 4. LIMITATIONS

**Known error risks:**
* May miss high-performers being headhunted externally (no market signal in data).
* Underperforms for very short-tenure employees (< 3 months).
* Departure reason predictions are indicative (confidence sometimes < 50%) due to small sample size (~104).
* Model sees a static snapshot — cannot detect sudden life events (divorce, illness, competing offer).

**Out-of-scope situations:**
* Employees from other companies or sectors.
* New hires with < 1 month of data.
* Employees on long-term sick leave.
* Any automated decision without human review.

**Bias risks:**
* Historical HR data may encode past management biases.
* `RecruitmentSource` is a top predictor — could proxy for demographics.
* Small dataset (311) — confidence intervals are wide; individual predictions should not be certainties.

---

## 5. RISKS & MITIGATION

**Misuse risks:**
* Over-interpreting a probability score as a definitive verdict.
* Using the tool to justify termination rather than retention.
* Applying the model to populations outside the training scope.

**Controls in place:**
* Output labelled **"DECISION SUPPORT ONLY — human review mandatory"**.
* Risk thresholds: Medium ≥ 40%, High ≥ 70% — not binary verdicts.
* Top 3 SHAP factors shown per prediction to prevent black-box use.
* Fairness audit re-run every time the model is retrained.
* Data poisoning test: AUC drop < 0.05 under 5% label corruption.
* Race/ethnicity columns dropped at ingestion.

---

## 6. ENERGY & FRUGALITY

* **Model size:** < 5 MB (Random Forest, 100 trees, max_depth=8)
* **Inference time:** < 1 ms per employee on CPU (negligible)
* **Training CO₂:** ~0.000001 kg CO₂eq (equivalent to 0.003 seconds of phone charging)

**Frugal choices made:**
* Lightweight models tested first (LR, DT) before ensembles.
* No neural network used — unjustified for 311 rows.
* `SHAP TreeExplainer` used (exact, fast) vs KernelExplainer.

---

## 7. CYBERSECURITY & COMPLIANCE

**Input validation:**
* All PII removed before processing (hash + drop pipeline).
* Numeric feature values clipped to [0, 99th percentile] to prevent adversarial outlier injection.
* Data poisoning test: PASSED (AUC degrades < 0.05 under 5% label corruption).

**Secrets management:**
* Hash salt (`HASH_SALT`) stored as environment variable — never hardcoded.
* No API keys or credentials present in the notebook.
* Dataset loaded from local path only — no external API calls.

**AI Act compliance:**
* Classified as **HIGH RISK** (Annex III §4 — employment systems).
* All mandatory safeguards documented: transparency, human oversight, fairness audit, robustness testing.
* GDPR compliance block present in the notebook.
