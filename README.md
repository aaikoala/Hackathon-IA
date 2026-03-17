# Hackathon-IA : Employee Attrition Predictor
### Hackathon AI x HR — Capgemini

> Predict which employees are at risk of leaving — and explain why — using responsible AI.

---

## Table of Contents
1. [Objectives](#objectives)
2. [Scope](#scope)
3. [Personas](#personas)
4. [Responsible AI](#responsible-ai)
5. [Instructions](#instructions)
6. [Deliverables](#deliverables)
7. [Dependencies](#dependencies)

---

## Objectives

This project builds an end-to-end AI solution to help HR management:

- **Predict** which employees are at risk of resigning.
- **Explain** why, using interpretable SHAP feature attributions.
- **Recommend** targeted HR actions per at-risk employee.

Everything is built in accordance with four responsible AI principles:
- **Data protection & cybersecurity**: Drop identifying columns, pseudonymise IDs, GDPR-compliant pipeline.
- **Ethics & non-discrimination**: Exclude sensitive attributes from model features, run fairness audit.
- **Frugality**: scikit-learn GradientBoosting — CPU only, < 5 MB, < 15s training.
- **Explainability**: SHAP TreeExplainer — every prediction comes with its top 3 drivers.

---

## Scope

### In scope
- Binary classification: `Termd = 0` (active) vs `Termd = 1` (resigned / terminated).
- Structured HR data from HRDataset_v14 (311 employees, 36 variables).
- Text enrichment via rule-based templates (exit interviews, satisfaction surveys, transfer requests).
- SHAP-based global and per-employee explainability.
- Fairness audit by `Sex` and `RaceDesc`.
- HR dashboard insights: top 10 at-risk employees + recommended actions.

### Out of scope
- Real-time scoring API / production deployment.
- Deep learning or large language model fine-tuning.
- Data from outside the provided dataset.

---

## Personas

### Persona 1 — The HR Director (End User)
> *"I need to understand which of my employees might leave in the next 3 months, and I need a clear reason — not a black box."*
- **Needs**: Actionable alerts, plain-language explanations, no false accusation risk.
- **Pain points**: High turnover costs, reactive rather than proactive HR.
- **Expects**: A prioritised list of at-risk employees with suggested actions.

### Persona 2 — The Data Scientist / Developer (Builder)
> *"I need to build a fair, frugal, explainable model on sensitive HR data and make sure it doesn't discriminate."*
- **Needs**: Clean pipeline, reproducible notebook, documented model choices.
- **Pain points**: Library conflicts, class imbalance, sensitive data handling.
- **Expects**: Working code, model card, data card.

### Persona 3 — The Company (Client)
> *"We are facing a high turnover rate. We want an AI solution that helps us retain talent — ethically and transparently."*
- **Needs**: Business ROI, legal compliance (GDPR, AI Act), trust in results.
- **Pain points**: Past HR decisions were reactive and sometimes unfair.
- **Expects**: A solution they can present to employees and regulators.

---

## Responsible AI

### GDPR
- Direct identifiers (`Employee_Name`, `DOB`, `Zip`, `State`, `ManagerName`) are **dropped** before any processing.
- `EmpID` is **pseudonymised** to `EMP_XXXX`.
- Sensitive attributes (`Sex`, `RaceDesc`, `HispanicLatino`, `CitizenDesc`, `MaritalDesc`) are **excluded from model features** and kept only for the fairness audit.

### Ethics & Fairness
- Model does **not use** gender, ethnicity, marital status, or citizenship as predictive features.
- A dedicated fairness audit cell checks prediction rates across demographic groups.
- A gap > 20% between groups signals a potential bias to investigate.

### Explainability
- Every prediction includes its **top 3 SHAP contributors** in plain language.
- No score is ever shown to HR without an accompanying explanation.
- HR managers remain the **sole decision-makers** — the model is advisory only.

### Frugality
- `GradientBoostingClassifier` from scikit-learn: no external compiled dependencies.
- Runs on CPU — no GPU required.
- Model size: ~2–5 MB | Training time: < 15 s | Inference: < 1 ms per employee.

### AI Act
- Classification: **High risk** (employment context).
- Requirements met: human oversight, transparency, decision traceability, fairness audit.

## Instructions
