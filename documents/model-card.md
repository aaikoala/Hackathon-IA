## Employee Attrition Predictor


This repository contains the Employee Attrition Predictor, a Gradient Boosting Classifier equipped with SHAP explainability[cite: 2, 3].
It is designed strictly as a decision-support tool for HR managers and CHROs to predict employee resignation risk[cite: 6].

---

### Model Overview

* **Task:** Binary classification to predict employee resignation risk.
* **Inputs:** Structured HR variables including salary, engagement, satisfaction, absences, late days, tenure, department, performance score, and special projects count.
* **Outputs:** A resignation probability score (0-1), a predicted class, and the top 3 SHAP drivers explained in plain language.
* **Architecture:** Native sklearn Gradient Boosting requiring no GPU (100% CPU).

---

### Training Data 

* **Dataset:** Built on the open-source, synthetic HRDataset_v14 from Kaggle.
* **Size:** Contains 311 employees with 36 structured variables.
* **Class Distribution:** Originally 207 active (67%) and 104 resigned (33%), rebalanced using SMOTE oversampling.
* **Text Enrichment:** Features three generated text columns (exit feedback, satisfaction survey, transfer request) that align coherently with employee metrics.
* **Privacy:** Direct identifiers (Employee_Name, DOB, Zip, State, ManagerName) were dropped for GDPR compliance.

---

### Performance Metrics

* **AUC-ROC:** 0.87, indicating excellent class separation.
* **Cross-val AUC (5-fold):** 0.85 +/- 0.04, confirming model robustness.
* **Precision (Class 1):** 0.81, meaning 81% of alerts are genuinely at risk.
* **Recall (Class 1):** 0.74.
* **Metric Strategy:** Recall is deliberately favoured over precision, as missing a true leaver (false negative) is more costly in an HR context than a false alarm.

---

### Limitations & Risks

* **Causation vs. Correlation:** The model identifies correlations, not causes; a high score does not imply the employee will leave.
* **Out-of-Distribution Data:** Poor performance is expected on highly atypical profiles (salary > $200k, tenure < 1 month).
* **Temporal Drift:** HR dynamics evolve, so retraining is recommended at a minimum of every 6 months.
* **Decision Safeguards:** Every prediction includes the top 3 SHAP drivers, and the score is never shown without explanation.
* **Human Oversight:** This is a decision-support tool only, requiring mandatory HR validation before any action is taken.

---

### Security, Compliance & Frugality

* **Cybersecurity:** Format and value-range checks are performed on every feature before inference, rejecting malformed inputs.
* **Anti-Injection:** No free-text is executed as code; text fields are treated as data only.
* **Compliance:** Classified as a High risk AI system under the AI Act, meeting enhanced requirements for transparency, human oversight, and decision traceability.
* **Footprint:** The serialised model (.pkl file) is roughly 2-5 MB.
* **Energy Efficiency:** Training takes less than 15 seconds on a standard CPU, inference takes under 1 ms per employee, and a full training run uses an estimated < 0.001 kWh.