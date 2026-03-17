# Data Card: HR Dataset v14.5

**Source:** Kaggle — Rich Huebner Human Resources Data Set  
**Note:** Fictional dataset — no real individuals.

---

## 1. Dataset Overview
- **Rows:** ~300 employees (historical and current)
- **Columns (raw):** 36
- **Target Variable:** `Termd` (Binary classification: `1` = terminated/resigned, `0` = active)

## 2. Features Used in Model
**Numeric Features:**
- `Salary`, `Age`, `EngagementSurvey`, `EmpSatisfaction`, `SpecialProjectsCount`, `DaysLateLast30`, `Absences`, `PerfScoreID`

**Categorical Features:**
- `Department`, `Position`, `State`, `RecruitmentSource`, `PerformanceScore`, `MaritalDesc`, `CitizenDesc`

## 3. Sensitive Attributes (Fairness Audit Only)
*These features are strictly excluded from model inputs to prevent algorithmic bias, and are only used post-prediction for fairness evaluation.*
- **Sex (M/F):** Used for demographic parity audit.
- **RaceDesc (Ethnicity):** Used for equalized odds audit.
- **HispanicLatino (Yes/No):** Audit attribute.

## 4. Preprocessing Pipeline
1. **PII Removal:** `Employee_Name`, `SSN`, `DOB`, `EmpID`, and `ManagerName` are hard-dropped.
2. **Leakage Prevention:** Target-leaking variables such as `TermReason` and `DateofTermination` are strictly excluded.
3. **Missing Values:** Addressed via median imputation for numeric features.
4. **Encoding:** `LabelEncoder` applied to categorical features with cardinality <= 20.
5. **Train/Test Split:** 80/20 ratio, stratified on the target variable (`Termd`).
6. **Scaling:** `StandardScaler` applied exclusively for the Logistic Regression model (fitted only on the training set to prevent data leakage).

## 5. GDPR Compliance & Privacy
- **Legal Basis:** Processed under Legitimate Interest (Art. 6(1)(f)).
- **Special Categories:** Attributes like `Sex` and `RaceDesc` are processed under Art. 9 exceptions solely for the purpose of algorithmic bias auditing.
- **Data Retention:** Only fully anonymised and pseudonymised data is stored in the pipeline.
- **Data Subject Rights:** The Right to Erasure applies to the original source records prior to anonymisation.

## 6. Data Quality & Limitations
- **Synthetic Data:** The dataset is illustrative and does not represent a real workforce; some correlations may be artificial.
- **Class Imbalance:** The termination rate is approximately 33%. This imbalance is actively handled during modeling (e.g., via `class_weight` parameters).
- **Granularity:** The `Salary` band is broad; highly granular pay data is unavailable for fine-tuning.
