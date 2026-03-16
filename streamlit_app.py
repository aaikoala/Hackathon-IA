"""
HR Retention Advisor — Streamlit Dashboard
Run with: streamlit run streamlit_app.py
"""

import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HR Retention Advisor",
    page_icon="👥",
    layout="wide",
)

ARTIFACT_PATH = Path(__file__).parent / "outputs" / "model_artifacts.pkl"

# ── Load artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open(ARTIFACT_PATH, "rb") as f:
        return pickle.load(f)

try:
    art = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model artifacts not found. "
        "Run the **Save Model Artifacts** cell in Hackathon_code.ipynb first."
    )
    st.stop()

rf_model            = art["rf_model"]
reason_model        = art["reason_model"]
explainer           = art["explainer"]
FEATURE_COLS        = art["feature_cols"]
MEDIANS             = art["feature_medians"]
MINS                = art["feature_mins"]
MAXS                = art["feature_maxs"]
REASON_FEATURE_COLS = art["reason_feature_cols"]

# ── Recommendation engine (mirrors the notebook) ───────────────────────────────
REASON_ACTIONS = {
    "Voluntary - Better Opportunity": {
        "icon": "🚀", "color": "#e67e22",
        "summary": "Likely leaving for a better external offer.",
        "actions": [
            "Conduct an internal mobility review — is there a promotion path available?",
            "Benchmark salary and title against market rate for this role.",
            "Have a candid career conversation: where do they want to be in 2 years?",
        ],
    },
    "Voluntary - Compensation": {
        "icon": "💰", "color": "#e74c3c",
        "summary": "Likely leaving primarily for higher pay.",
        "actions": [
            "Fast-track a compensation review with their manager and HR BP.",
            "Explore non-salary benefits: bonus, equity, extra leave.",
            "Check if the role is underbanded compared to new hires.",
        ],
    },
    "Voluntary - Dissatisfaction": {
        "icon": "😔", "color": "#c0392b",
        "summary": "Likely leaving due to unhappiness with role, team, or culture.",
        "actions": [
            "Schedule a confidential 1-on-1 — focus on listening, not defending.",
            "Investigate team dynamics: manager relationship, peer conflicts.",
            "Explore an internal transfer to a different team or project.",
        ],
    },
    "Voluntary - Work Conditions": {
        "icon": "⏰", "color": "#8e44ad",
        "summary": "Likely leaving due to workload, hours, or work-life balance.",
        "actions": [
            "Review workload distribution within the team.",
            "Offer flexible/hybrid schedule or reduced hours temporarily.",
            "Explore a temporary leave of absence instead of full resignation.",
        ],
    },
    "Voluntary - Personal": {
        "icon": "🏠", "color": "#2980b9",
        "summary": "Likely leaving for personal reasons (relocation, studies, family).",
        "actions": [
            "Offer remote work if the role allows — removes relocation barrier.",
            "Discuss a leave of absence or part-time arrangement.",
            "Keep the door open: document as eligible for rehire.",
        ],
    },
    "Involuntary - Performance/Conduct": {
        "icon": "⚠️", "color": "#d35400",
        "summary": "At risk of termination for performance or conduct issues.",
        "actions": [
            "Launch a formal Performance Improvement Plan (PIP) with clear milestones.",
            "Increase check-in frequency with line manager (weekly 1-on-1).",
            "Investigate root cause: skills gap, personal issue, or disengagement?",
        ],
    },
    "Health / Personal": {
        "icon": "🏥", "color": "#16a085",
        "summary": "May be leaving due to health or personal circumstances.",
        "actions": [
            "Refer to Employee Assistance Programme (EAP) for confidential support.",
            "Explore medical leave, flexible hours, or temporary reduced duties.",
            "Ensure manager is aware of reasonable adjustment obligations.",
        ],
    },
    "Retirement": {
        "icon": "🎓", "color": "#27ae60",
        "summary": "Approaching retirement age.",
        "actions": [
            "Begin knowledge transfer planning — document critical expertise.",
            "Explore phased retirement or part-time consultancy arrangement.",
            "Identify and start developing an internal successor.",
        ],
    },
    "Other": {
        "icon": "❓", "color": "#7f8c8d",
        "summary": "Departure reason unclear — requires direct conversation.",
        "actions": [
            "Schedule a stay interview to understand current satisfaction.",
            "Review the full employee profile with their line manager.",
        ],
    },
}

SHAP_RECOMMENDATIONS = {
    "absences":            ("high", "High absenteeism — possible burnout or disengagement.",
                            "Schedule a wellness check-in; consider flexible work arrangements."),
    "empsatisfaction":     ("low",  "Low satisfaction score.",
                            "Arrange a 1-on-1 to surface pain points; review workload and role fit."),
    "engagementsurvey":    ("low",  "Low engagement survey result.",
                            "Offer stretch assignments or cross-team projects; explore mentoring."),
    "dayslateLast30":      ("high", "Frequent late arrivals.",
                            "Discuss commute or scheduling issues; consider flexible start times."),
    "salary":              ("low",  "Salary below peers in similar roles.",
                            "Conduct a pay equity analysis; fast-track a compensation review."),
    "specialprojectscount":("low",  "Few or no special projects.",
                            "Assign a visible project; connect to a sponsor or mentor."),
    "perfscoreid":         ("low",  "Below-average performance score.",
                            "Set up a structured development plan with regular coaching."),
}


def get_shap_recommendation(feature: str, shap_val: float) -> tuple[str, str]:
    key = feature.lower()
    direction = "high" if shap_val > 0 else "low"
    for feat_key, (dir_key, reason, action) in SHAP_RECOMMENDATIONS.items():
        if feat_key in key and dir_key == direction:
            return reason, action
    dir_word = "high" if shap_val > 0 else "low"
    return f"{dir_word} value for {feature}", f"Review this metric with the line manager."


# ── Build input row from user inputs ──────────────────────────────────────────
def build_feature_row(inputs: dict) -> pd.DataFrame:
    """Fill all model features with median defaults, override with user inputs."""
    row = {col: MEDIANS.get(col, 0.0) for col in FEATURE_COLS}
    for key, val in inputs.items():
        if key in row:
            row[key] = float(val)
    return pd.DataFrame([row])[FEATURE_COLS]


# ── Predict ───────────────────────────────────────────────────────────────────
def predict(feature_row: pd.DataFrame) -> dict:
    prob       = float(rf_model.predict_proba(feature_row)[0, 1])
    sv         = explainer.shap_values(feature_row)

    if isinstance(sv, list):
        sv_pos = sv[1][0]
    elif sv.ndim == 3:
        sv_pos = sv[0, :, 1]
    else:
        sv_pos = sv[0]

    shap_series = pd.Series(sv_pos, index=FEATURE_COLS)

    # Departure reason
    feat_reason = feature_row.iloc[0].reindex(REASON_FEATURE_COLS, fill_value=0)
    reason_proba = reason_model.predict_proba(feat_reason.values.reshape(1, -1))[0]
    top_idx      = int(np.argmax(reason_proba))
    category     = reason_model.classes_[top_idx]
    confidence   = float(reason_proba[top_idx])

    return {
        "prob":       prob,
        "shap":       shap_series,
        "category":   category,
        "confidence": confidence,
        "all_reasons": sorted(
            zip(reason_model.classes_, reason_proba), key=lambda x: -x[1]
        ),
    }


# =============================================================================
#  UI
# =============================================================================

st.title("👥 HR Retention Advisor")
st.caption(
    "Enter employee information to assess turnover risk and get tailored HR recommendations. "
    "All inputs are anonymised — no personal identifiers are used."
)
st.divider()

# ── Sidebar — employee form ───────────────────────────────────────────────────
with st.sidebar:
    st.header("📋 Employee Profile")
    st.caption("Fill in what you know. Other fields default to company averages.")

    st.subheader("💼 Compensation & Role")
    salary = st.number_input(
        "Annual Salary (USD)", min_value=30000, max_value=250000,
        value=int(MEDIANS.get("Salary", 60000)), step=1000,
    )
    perf_label = st.selectbox(
        "Performance Score",
        ["PIP (1)", "Needs Improvement (2)", "Fully Meets (3)", "Exceeds (4)"],
        index=2,
    )
    perf_score = int(perf_label[perf_label.rfind("(")+1])

    special_projects = st.slider(
        "Special Projects Count", 0, 10,
        value=int(MEDIANS.get("SpecialProjectsCount", 1)),
    )

    st.subheader("📊 Engagement Metrics")
    satisfaction = st.slider(
        "Employee Satisfaction (1=low → 5=high)", 1, 5,
        value=int(MEDIANS.get("EmpSatisfaction", 3)),
    )
    engagement = st.slider(
        "Engagement Survey Score (1.0 → 5.0)", 1.0, 5.0,
        value=float(MEDIANS.get("EngagementSurvey", 3.5)), step=0.1,
    )

    st.subheader("🚨 Behavioural Signals")
    absences = st.number_input(
        "Absences (days/year)", min_value=0, max_value=100,
        value=int(MEDIANS.get("Absences", 8)),
    )
    days_late = st.number_input(
        "Days Late (last 30 days)", min_value=0, max_value=30,
        value=int(MEDIANS.get("DaysLateLast30", 0)),
    )

    st.divider()
    analyse = st.button("🔍 Analyse Employee", use_container_width=True, type="primary")

# ── Main panel ────────────────────────────────────────────────────────────────
if not analyse:
    st.info(
        "👈 Fill in the employee profile in the sidebar and click **Analyse Employee**."
    )
    st.markdown(
        """
        ### How to use this tool
        1. Enter the employee's key metrics in the **sidebar**
        2. Click **Analyse Employee**
        3. Review the **risk score**, predicted **departure reason**, and **recommended actions**

        > ⚠️ **This tool is decision-support only.** All predictions must be reviewed by
        > a qualified HR professional before any action is taken.
        """
    )
    st.stop()

# ── Run prediction ─────────────────────────────────────────────────────────────
user_inputs = {
    "Salary":               salary,
    "EmpSatisfaction":      satisfaction,
    "EngagementSurvey":     engagement,
    "Absences":             absences,
    "DaysLateLast30":       days_late,
    "SpecialProjectsCount": special_projects,
    "PerfScoreID":          perf_score,
}

feature_row = build_feature_row(user_inputs)
result      = predict(feature_row)

prob       = result["prob"]
shap_vals  = result["shap"]
category   = result["category"]
confidence = result["confidence"]
reason_info = REASON_ACTIONS.get(category, REASON_ACTIONS["Other"])

# ── Risk level ────────────────────────────────────────────────────────────────
if prob >= 0.70:
    risk_level = "HIGH RISK"
    risk_color = "#e74c3c"
    risk_emoji = "🔴"
elif prob >= 0.40:
    risk_level = "MEDIUM RISK"
    risk_color = "#e67e22"
    risk_emoji = "🟠"
else:
    risk_level = "LOW RISK"
    risk_color = "#2ecc71"
    risk_emoji = "🟢"

# ── Layout: 3 top metrics ──────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"""
        <div style="background:{risk_color}22; border-left:5px solid {risk_color};
                    padding:16px; border-radius:8px; text-align:center;">
            <div style="font-size:2.5rem; font-weight:bold; color:{risk_color}">
                {prob:.0%}
            </div>
            <div style="font-size:1rem; color:{risk_color}; font-weight:600">
                {risk_emoji} {risk_level}
            </div>
            <div style="font-size:0.8rem; color:#666; margin-top:4px">
                Predicted turnover probability
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    cat_color = reason_info["color"]
    st.markdown(
        f"""
        <div style="background:{cat_color}22; border-left:5px solid {cat_color};
                    padding:16px; border-radius:8px; text-align:center;">
            <div style="font-size:2rem">{reason_info['icon']}</div>
            <div style="font-size:0.95rem; font-weight:600; color:{cat_color}">
                {category}
            </div>
            <div style="font-size:0.8rem; color:#666; margin-top:4px">
                Predicted departure reason ({confidence:.0%} confidence)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    top3 = shap_vals.abs().sort_values(ascending=False).head(3)
    top_feat = top3.index[0] if len(top3) > 0 else "N/A"
    st.markdown(
        f"""
        <div style="background:#3498db22; border-left:5px solid #3498db;
                    padding:16px; border-radius:8px; text-align:center;">
            <div style="font-size:1.5rem; font-weight:bold; color:#3498db">
                {top_feat}
            </div>
            <div style="font-size:0.95rem; font-weight:600; color:#3498db">
                Top risk driver
            </div>
            <div style="font-size:0.8rem; color:#666; margin-top:4px">
                Strongest SHAP signal
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# ── Two columns: SHAP chart | Reasons breakdown ───────────────────────────────
left, right = st.columns([1.2, 1])

with left:
    st.subheader("📊 Risk Factors (SHAP)")
    st.caption("Bars pushing right → increase risk. Bars pushing left → decrease risk.")

    top_n = shap_vals.abs().sort_values(ascending=False).head(8)
    top_shap = shap_vals[top_n.index].sort_values()

    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in top_shap.values]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.barh(top_shap.index, top_shap.values, color=colors, edgecolor="black", alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on turnover risk)")
    ax.set_title("Feature Contributions", fontsize=11)
    for bar, val in zip(bars, top_shap.values):
        ax.text(
            val + (0.002 if val >= 0 else -0.002),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}",
            va="center", ha="left" if val >= 0 else "right", fontsize=8,
        )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with right:
    st.subheader("🎯 Departure Reason Probabilities")
    st.caption("How likely each departure category is for this employee.")

    reason_df = pd.DataFrame(result["all_reasons"], columns=["Category", "Probability"])
    reason_df = reason_df[reason_df["Probability"] > 0.01].sort_values("Probability", ascending=False)

    for _, row in reason_df.iterrows():
        cat    = row["Category"]
        p      = row["Probability"]
        r_info = REASON_ACTIONS.get(cat, REASON_ACTIONS["Other"])
        bar_w  = int(p * 100)
        st.markdown(
            f"""
            <div style="margin-bottom:8px">
                <div style="display:flex; justify-content:space-between;
                            font-size:0.85rem; margin-bottom:2px">
                    <span>{r_info['icon']} {cat}</span>
                    <span style="font-weight:bold">{p:.0%}</span>
                </div>
                <div style="background:#eee; border-radius:4px; height:8px">
                    <div style="background:{r_info['color']}; width:{bar_w}%;
                                height:8px; border-radius:4px"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()

# ── Recommended HR actions ────────────────────────────────────────────────────
st.subheader("✅ Recommended HR Actions")

tab1, tab2 = st.tabs(
    [f"{reason_info['icon']} Based on predicted reason", "🔍 Based on risk signals (SHAP)"]
)

with tab1:
    st.info(f"**{category}** — {reason_info['summary']}")
    for i, action in enumerate(reason_info["actions"], 1):
        st.markdown(f"**{i}.** {action}")

with tab2:
    top3_features = shap_vals.abs().sort_values(ascending=False).head(3)
    for feat in top3_features.index:
        val        = float(feature_row.iloc[0][feat])
        shap_v     = float(shap_vals[feat])
        reason, action = get_shap_recommendation(feat, shap_v)
        direction  = "▲ high" if shap_v > 0 else "▼ low"
        with st.expander(f"**{feat}** = {val:.1f}  [{direction}]  →  {reason}"):
            st.markdown(f"💡 {action}")

st.divider()

# ── Plain-English summary ─────────────────────────────────────────────────────
st.subheader("📝 Plain-English Summary")
top2 = shap_vals.abs().sort_values(ascending=False).head(2).index.tolist()
r1, _ = get_shap_recommendation(top2[0], float(shap_vals[top2[0]]))
r2, _ = get_shap_recommendation(top2[1], float(shap_vals[top2[1]])) if len(top2) > 1 else ("", "")

summary = (
    f"This employee has a **{prob:.0%} predicted probability of leaving** ({risk_level}). "
    f"The most likely reason is **{category}** (confidence: {confidence:.0%}). "
    f"The two strongest risk signals are **{r1}** and **{r2}**."
)
st.markdown(
    f"""
    <div style="background:#f8f9fa; border:1px solid #dee2e6;
                border-radius:8px; padding:16px; font-size:1rem; line-height:1.6">
        {summary}
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "⚠️ GDPR Notice: No personal identifiers are processed. "
    "This tool is decision-support only — all HR actions require human review. "
    "EU AI Act classification: HIGH RISK (Annex III §4)."
)
