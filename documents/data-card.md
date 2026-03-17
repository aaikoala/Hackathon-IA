# Data Card: HR Attrition Dataset

## 1. Dataset Overview
- **Name:** HRDataset_v14
- **Description:** A structured Human Resources dataset containing employee records, performance reviews, and demographic information.
- **Primary Use Case:** Binary classification of employee attrition (`Termd` = 0 or 1).
- **Size:** 311 rows (employees) × 36 initial columns.

## 2. Provenance & Collection
- **Source:** The original dataset is *HRDataset_v14*, a widely used synthetic HR dataset created by Dr. Rich Huebner for educational purposes.

## 3. Composition & Key Variables
- **Target Variable:** `Termd` (Integer) 
  - `0`: Active employee.
  - `1`: Terminated or resigned employee.
- **Key Predictive Features:**
  - `EngagementSurvey`: Employee engagement score (1.0 to 5.0).
  - `EmpSatisfaction`: Job satisfaction level (1 to 5).
  - `Salary`: Annual salary in USD.
  - `Absences`: Number of absent days.
  - `SpecialProjectsCount`: Number of special projects the employee is involved in.

## 4. Data Privacy & GDPR Compliance
To adhere to data minimization and privacy-by-design principles, the following transformations are applied before any model training:
- **Direct Identifiers Removed:** `Employee_Name`, `DOB` (Date of Birth), `Zip`, `State`, and `ManagerName` are hard-dropped.
- **Pseudonymisation:** The original `EmpID` is masked or replaced with a standard format (`EMP_XXXX`).

## 5. Sensitive Data & Fairness
The dataset contains protected demographic attributes. To comply with ethical guidelines and French Labor Law (Art. L1132-1):
- **Excluded from Model:** `Sex`, `RaceDesc`, `HispanicLatino`, `CitizenDesc`, and `MaritalDesc` are **strictly excluded** from the feature set. The model cannot use these to make predictions.

## 6. Known Limitations
- **Size:** With only 311 records, the dataset is relatively small, which can lead to overfitting if complex models are used (hence the choice of frugal models like Decision Trees or Gradient Boosting).
- **Class Imbalance:** The dataset is imbalanced (approximately 1/3 turnover rate). Metrics like Accuracy are less reliable than AUC-ROC or F1-Score for evaluating model performance.
- **Synthetic Nature:** As the data is synthetic, some correlations might be overly clean compared to noisy, real-world HR data.
