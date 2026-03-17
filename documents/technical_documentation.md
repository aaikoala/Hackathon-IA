# 📄 Technical Documentation:  Predictive Model

## 1. Project Overview
This project implements a Machine Learning pipeline aimed at predicting employee turnover risk at LuminaTech. It is designed around two major constraints: **strict compliance with GDPR and French Labor Law**, and the principles of **Frugal AI** (low carbon footprint, low compute cost, and high explainability).

## 2. Environment & Dependencies
*(List the necessary components to run your `Hackathon_code.ipynb` notebook here)*
* **Python version:** 3.8+ (recommended)
* **Core Libraries:** `pandas`, `numpy` (Data manipulation)
* **Machine Learning:** `scikit-learn` (Modeling and metrics)
* **Data Visualization:** `matplotlib`, `seaborn` (Graph generation)
* **Green IT / Profiling:** *(Specify here if you used a specific library like `codecarbon`, `eco2AI`, or Python's native `time` module to measure the training time and CO2 footprint shown on slide 4).*

## 3. Data Processing & Privacy by Design
This section details the transformation of the raw dataset (311 rows, 36 columns) into the training dataset (25 columns).

* **Data Ingestion:** Loading the source file (e.g., `HRDataset_v14.csv`).
* **Anonymization (GDPR Compliance):** Hard-dropping direct identifiers to ensure privacy by design.
    * *Columns dropped:* `Employee_Name`, `DOB`, `Zip`.
* **Legal Compliance (French Labor Law Art. L1132-1):** Removing sensitive and potentially discriminatory variables to prevent algorithmic bias and strictly comply with national law.
    * *Columns dropped:* `Sex`, `RaceDesc`.
* **Feature Engineering & Encoding:** *(Specify your methods here: Did you use `pd.get_dummies` or `LabelEncoder` for categorical variables like Department or Role? How did you handle missing values?)*

## 4. Modeling Pipeline
Explanation of the training and evaluation strategy.

* **Train/Test Split:** *(e.g., 80% train / 20% test split. Specify if you used `stratify=y` to handle the class imbalance visible in the turnover distribution chart).*
* **Models Evaluated:**
    1. Logistic Regression
    2. Random Forest
    3. **Decision Tree (Selected Model)**
    4. Gradient Boosting
    5. Support Vector Machine (SVM)
* **Hyperparameters:** *(Briefly indicate if you used scikit-learn's default parameters or performed a grid search (`GridSearchCV`) to optimize them).*

## 5. Evaluation & Frugal AI Metrics
This is the core of the responsible AI approach.

* **Predictive Performance:** Using **AUC (Area Under the ROC Curve)** to evaluate the model's ability to distinguish between employees who stay and those who leave, accounting for class imbalance.
* **Environmental & Compute Impact:**
    * *Train Time:* Measured in seconds and normalized (0-1 scale) for comparison.
    * *Carbon Footprint (CO2):* Estimated based on compute time and normalized (0-1 scale).
* **Explainability Trade-off:** Justification for selecting the **Decision Tree**. Although it has a slightly "Lower" AUC than the Random Forest, it boasts the "Lowest" compute/carbon cost and provides highly interpretable decision rules (e.g., "If salary < X and absences > Y, then high risk"). This white-box approach is mandatory for actionable HR insights.

## 6. How to Run the Code
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run all cells in the `Hackathon_code.ipynb` notebook.
