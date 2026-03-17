"""
HR Retention Advisor — Streamlit Dashboard
Run: C:\\Users\\valen\\anaconda3\\python.exe -m streamlit run streamlit_app.py
"""

import os
import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HR·AI Retention Advisor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 28px 32px;
        border-radius: 16px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        gap: 20px;
    }
    .logo-circle {
        width: 64px; height: 64px;
        background: linear-gradient(135deg, #e94560, #0f3460);
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 28px; flex-shrink: 0;
        box-shadow: 0 4px 15px rgba(233,69,96,0.4);
    }
    .header-text h1 {
        color: white; font-size: 1.9rem; font-weight: 700;
        margin: 0; letter-spacing: -0.5px;
    }
    .header-text p {
        color: #a0aec0; font-size: 0.9rem; margin: 4px 0 0 0;
    }
    .header-badges { display:flex; gap:8px; margin-top:8px; }
    .badge {
        background: rgba(255,255,255,0.1); color: #e2e8f0;
        padding: 3px 10px; border-radius: 20px; font-size: 0.72rem;
        border: 1px solid rgba(255,255,255,0.15);
    }

    .risk-card {
        padding: 20px; border-radius: 12px;
        text-align: center; height: 100%;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    }
    .metric-big { font-size: 2.8rem; font-weight: 800; line-height: 1; }
    .metric-label { font-size: 0.8rem; color: #718096; margin-top: 6px; }
    .metric-sub { font-size: 1rem; font-weight: 600; margin-top: 4px; }

    .section-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }

    .action-item {
        background: #f7fafc;
        border-left: 4px solid #4299e1;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin-bottom: 10px;
        font-size: 0.9rem;
        line-height: 1.5;
        color: #2d3748 !important;
    }
    .action-num {
        display: inline-block;
        background: #4299e1; color: white;
        border-radius: 50%; width: 22px; height: 22px;
        text-align: center; line-height: 22px;
        font-size: 0.75rem; font-weight: 700;
        margin-right: 8px;
    }

    .ai-box {
        background: linear-gradient(135deg, #667eea22, #764ba222);
        border: 1px solid #667eea55;
        border-radius: 12px;
        padding: 20px;
        position: relative;
    }
    .ai-badge {
        position: absolute; top: -10px; left: 16px;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white; padding: 2px 12px; border-radius: 20px;
        font-size: 0.72rem; font-weight: 600;
    }

    .shap-positive { color: #e53e3e; font-weight: 600; }
    .shap-negative { color: #38a169; font-weight: 600; }

    .reason-bar-wrap { margin-bottom: 10px; }
    .reason-bar-label {
        display: flex; justify-content: space-between;
        font-size: 0.83rem; margin-bottom: 3px;
    }
    .reason-bar-bg {
        background: #edf2f7; border-radius: 6px; height: 10px; overflow: hidden;
    }

    div[data-testid="stSidebar"] {
        background: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    div[data-testid="stSidebar"] h2 {
        color: #2d3748; font-size: 1rem; font-weight: 700;
    }

    [data-testid="stButton"] button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 12px !important;
        box-shadow: 0 4px 15px rgba(102,126,234,0.4) !important;
        transition: all 0.2s !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Artifacts ──────────────────────────────────────────────────────────────────
ARTIFACT_PATH = Path(__file__).parent / "outputs" / "model_artifacts.pkl"

@st.cache_resource
def load_artifacts():
    with open(ARTIFACT_PATH, "rb") as f:
        return pickle.load(f)

try:
    art = load_artifacts()
except FileNotFoundError:
    st.error("Model artifacts not found. Run the **Save Model Artifacts** cell in Hackathon_code.ipynb first.")
    st.stop()

rf_model            = art["rf_model"]
reason_model        = art["reason_model"]
explainer           = art["explainer"]
FEATURE_COLS        = art["feature_cols"]
MEDIANS             = art["feature_medians"]
MINS                = art["feature_mins"]
MAXS                = art["feature_maxs"]
REASON_FEATURE_COLS = art["reason_feature_cols"]

# ── Departure reason metadata ──────────────────────────────────────────────────
REASON_ACTIONS = {
    "Voluntary - Better Opportunity": {
        "icon": "🚀", "color": "#e67e22",
        "summary": "Likely leaving for a better external role or career opportunity.",
        "actions": [
            "Conduct an internal mobility review — is there a promotion path available?",
            "Benchmark salary and title against market rate for this role.",
            "Have a candid career conversation: where do they want to be in 2 years?",
        ],
    },
    "Voluntary - Compensation": {
        "icon": "💰", "color": "#e74c3c",
        "summary": "Likely leaving primarily for higher pay elsewhere.",
        "actions": [
            "Fast-track a compensation review with their manager and HR BP.",
            "Explore non-salary benefits: bonus, equity, extra leave.",
            "Check if the role is underbanded compared to recent new hires.",
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
        "summary": "Likely leaving due to workload, hours, or work-life balance issues.",
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
            "Offer remote work if the role allows — removes the relocation barrier.",
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
        "summary": "Departure reason unclear — requires a direct conversation.",
        "actions": [
            "Schedule a stay interview to understand current satisfaction.",
            "Review the full employee profile with their line manager.",
        ],
    },
}

SHAP_LABELS = {
    "absences":             ("high", "High absenteeism",         "Schedule wellness check-in; consider flexible arrangements."),
    "empsatisfaction":      ("low",  "Low satisfaction score",   "Arrange 1-on-1; review workload, team dynamics, role fit."),
    "engagementsurvey":     ("low",  "Low engagement",           "Offer stretch projects or mentoring opportunities."),
    "dayslateLast30":       ("high", "Frequent late arrivals",   "Discuss scheduling or commute; consider flexible start times."),
    "dayslatelast30":       ("high", "Frequent late arrivals",   "Discuss scheduling or commute; consider flexible start times."),
    "salary":               ("low",  "Below-market salary",      "Conduct pay equity review; fast-track compensation adjustment."),
    "specialprojectscount": ("low",  "No special projects",      "Assign visible project; connect to mentor outside direct team."),
    "perfscoreid":          ("low",  "Low performance score",    "Set up development plan with clear milestones and coaching."),
    "marriedid":            ("high", "Personal life change",     "Offer EAP resources; review workload and travel requirements."),
    "recruitmentsource":    ("high", "Recruitment channel risk", "Review onboarding quality; strengthen internal mobility path."),
}

def get_shap_label(feature: str, shap_val: float):
    key = feature.lower()
    direction = "high" if shap_val > 0 else "low"
    for feat_key, (dir_key, reason, action) in SHAP_LABELS.items():
        if feat_key in key and dir_key == direction:
            return reason, action
    dir_word = "high" if shap_val > 0 else "low"
    return f"{dir_word} {feature}", f"Review this metric with the line manager."

# ── Prediction ─────────────────────────────────────────────────────────────────
def build_row(inputs: dict) -> pd.DataFrame:
    row = {col: MEDIANS.get(col, 0.0) for col in FEATURE_COLS}
    for k, v in inputs.items():
        if k in row:
            row[k] = float(v)
    return pd.DataFrame([row])[FEATURE_COLS]

def predict(feature_row: pd.DataFrame) -> dict:
    prob = float(rf_model.predict_proba(feature_row)[0, 1])
    sv   = explainer.shap_values(feature_row)
    if isinstance(sv, list):
        sv_pos = sv[1][0]
    elif sv.ndim == 3:
        sv_pos = sv[0, :, 1]
    else:
        sv_pos = sv[0]

    shap_s = pd.Series(sv_pos, index=FEATURE_COLS)

    feat_r       = feature_row.iloc[0].reindex(REASON_FEATURE_COLS, fill_value=0)
    reason_proba = reason_model.predict_proba(feat_r.values.reshape(1, -1))[0]
    top_idx      = int(np.argmax(reason_proba))
    category     = reason_model.classes_[top_idx]
    confidence   = float(reason_proba[top_idx])

    return {
        "prob":       prob,
        "shap":       shap_s,
        "category":   category,
        "confidence": confidence,
        "all_reasons": sorted(zip(reason_model.classes_, reason_proba), key=lambda x: -x[1]),
    }

# ── AI recommendation via Claude ───────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def generate_ai_recommendation(prob: float, category: str, confidence: float,
                                 top_factors: str, api_key: str) -> str:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        prompt = f"""You are a senior HR consultant and organizational psychologist.
An AI model has analysed an anonymous employee profile and produced the following findings:

- Predicted turnover probability: {prob:.0%}
- Most likely departure reason: {category} (model confidence: {confidence:.0%})
- Top risk signals from SHAP analysis: {top_factors}

Write a concise, professional retention recommendation in 3 short paragraphs:
1. Summary of the situation and urgency level
2. Two or three specific, actionable retention interventions tailored to the signals above
3. One sentence on what NOT to do (common HR mistake for this profile)

Tone: direct, empathetic, evidence-based. No bullet points. No headers. Plain English."""

        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    except Exception as e:
        return f"_(AI recommendation unavailable: {e})_"

# ══════════════════════════════════════════════════════════════════════════════
#  HEADER / LOGO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
  <div class="logo-circle">🧠</div>
  <div class="header-text">
    <h1>HR·AI Retention Advisor</h1>
    <p>Predict employee turnover risk · Explain the drivers · Recommend targeted actions</p>
    <div class="header-badges">
      <span class="badge">🔐 GDPR Compliant</span>
      <span class="badge">⚖️ Bias-Free Model</span>
      <span class="badge">🔍 SHAP Explainable</span>
      <span class="badge">🤖 AI Recommendations</span>
      <span class="badge">🇪🇺 EU AI Act — High Risk</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — Employee form
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📋 Employee Profile")
    st.caption("Fill in what you know. Unlisted fields default to company averages.")

    st.markdown("### 💼 Compensation & Role")
    salary = st.number_input(
        "Annual Salary (USD)", min_value=30_000, max_value=300_000,
        value=int(MEDIANS.get("Salary", 60_000)), step=1_000,
    )
    perf_label = st.selectbox(
        "Performance Score",
        ["1 — PIP (serious issues)", "2 — Needs Improvement",
         "3 — Fully Meets Expectations", "4 — Exceeds Expectations"],
        index=2,
    )
    perf_score = int(perf_label[0])
    special_projects = st.slider(
        "Special Projects Count", 0, 10,
        value=int(MEDIANS.get("SpecialProjectsCount", 1)),
    )

    st.markdown("### 📊 Engagement")
    satisfaction = st.slider(
        "Employee Satisfaction  (1 = very low, 5 = very high)", 1, 5,
        value=int(MEDIANS.get("EmpSatisfaction", 3)),
    )
    engagement = st.slider(
        "Engagement Survey Score  (1.0 → 5.0)", 1.0, 5.0,
        value=round(float(MEDIANS.get("EngagementSurvey", 3.5)), 1), step=0.1,
    )

    st.markdown("### 🚨 Behavioural Signals")
    absences = st.number_input(
        "Absences (days / year)", min_value=0, max_value=100,
        value=int(MEDIANS.get("Absences", 8)),
    )
    days_late = st.number_input(
        "Days Late (last 30 days)", min_value=0, max_value=30,
        value=int(MEDIANS.get("DaysLateLast30", 0)),
    )

    st.divider()
    analyse = st.button("🔍  Analyse Employee", use_container_width=True, type="primary")

# ── Landing page ───────────────────────────────────────────────────────────────
if not analyse:
    c1, c2, c3 = st.columns(3)
    for col, icon, title, desc in [
        (c1, "🎯", "Predict", "Assess turnover risk with a Random Forest model trained on HR data"),
        (c2, "🔍", "Explain", "SHAP waterfall chart shows exactly which factors drive the prediction"),
        (c3, "💡", "Act", "AI-generated, personalised retention plan based on the analysis"),
    ]:
        with col:
            st.markdown(f"""
            <div class="section-card" style="text-align:center; padding:28px">
                <div style="font-size:2.5rem">{icon}</div>
                <div style="font-size:1.1rem; font-weight:700; margin:8px 0 4px">{title}</div>
                <div style="font-size:0.85rem; color:#718096">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="section-card" style="background:#fffbeb; border-color:#f6e05e">
        <b>⚠️ Decision-support only.</b>
        All predictions must be reviewed by a qualified HR professional before any action is taken.
        No personal identifiers are processed. EU AI Act classification: <b>HIGH RISK</b> (Annex III §4).
    </div>""", unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
#  RUN PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
user_inputs = {
    "Salary": salary, "EmpSatisfaction": satisfaction,
    "EngagementSurvey": engagement, "Absences": absences,
    "DaysLateLast30": days_late, "SpecialProjectsCount": special_projects,
    "PerfScoreID": perf_score,
}
feature_row = build_row(user_inputs)
result      = predict(feature_row)

prob        = result["prob"]
shap_vals   = result["shap"]
category    = result["category"]
confidence  = result["confidence"]
reason_info = REASON_ACTIONS.get(category, REASON_ACTIONS["Other"])

if prob >= 0.70:
    risk_level, risk_color, risk_emoji = "HIGH RISK",   "#e53e3e", "🔴"
elif prob >= 0.40:
    risk_level, risk_color, risk_emoji = "MEDIUM RISK", "#dd6b20", "🟠"
else:
    risk_level, risk_color, risk_emoji = "LOW RISK",    "#38a169", "🟢"

top3_feats = shap_vals.abs().sort_values(ascending=False).head(3).index.tolist()

# ══════════════════════════════════════════════════════════════════════════════
#  TOP METRIC CARDS
# ══════════════════════════════════════════════════════════════════════════════
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
    <div class="risk-card" style="background:{risk_color}12; border:2px solid {risk_color}44">
        <div class="metric-big" style="color:{risk_color}">{prob:.0%}</div>
        <div class="metric-sub" style="color:{risk_color}">{risk_emoji} {risk_level}</div>
        <div class="metric-label">Predicted turnover probability</div>
    </div>""", unsafe_allow_html=True)

with c2:
    cat_color = reason_info["color"]
    st.markdown(f"""
    <div class="risk-card" style="background:{cat_color}12; border:2px solid {cat_color}44">
        <div style="font-size:2rem">{reason_info['icon']}</div>
        <div class="metric-sub" style="color:{cat_color}; font-size:0.9rem">{category}</div>
        <div class="metric-label">Predicted departure reason · {confidence:.0%} confidence</div>
    </div>""", unsafe_allow_html=True)

with c3:
    top1_label, _ = get_shap_label(top3_feats[0], float(shap_vals[top3_feats[0]]))
    st.markdown(f"""
    <div class="risk-card" style="background:#4299e112; border:2px solid #4299e144">
        <div style="font-size:1.3rem; font-weight:700; color:#2b6cb0">{top3_feats[0]}</div>
        <div class="metric-sub" style="color:#2b6cb0; font-size:0.85rem">{top1_label}</div>
        <div class="metric-label">Strongest SHAP signal</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SHAP WATERFALL  |  DEPARTURE REASON BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════════
left, right = st.columns([1.3, 1])

with left:
    st.markdown("#### 📊 SHAP Risk Factor Analysis")
    st.caption("🔴 Red bars increase turnover risk · 🟢 Green bars decrease it")

    top_n    = shap_vals.abs().sort_values(ascending=False).head(8)
    top_shap = shap_vals[top_n.index].sort_values()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor('#fafafa')
    ax.set_facecolor('#fafafa')

    colors = ["#e53e3e" if v > 0 else "#38a169" for v in top_shap.values]
    bars = ax.barh(
        [f.replace("_", " ") for f in top_shap.index],
        top_shap.values,
        color=colors, edgecolor="white", linewidth=0.8,
        height=0.65, alpha=0.9,
    )
    ax.axvline(0, color="#2d3748", linewidth=1.2, zorder=5)
    ax.set_xlabel("SHAP value  (impact on turnover probability)", fontsize=9, color="#4a5568")
    ax.set_title("Feature Contributions to Turnover Risk", fontsize=11,
                 fontweight="bold", color="#2d3748", pad=10)
    ax.tick_params(axis="y", labelsize=9, colors="#4a5568")
    ax.tick_params(axis="x", labelsize=8, colors="#718096")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#e2e8f0")

    for bar, val in zip(bars, top_shap.values):
        offset = 0.003 if val >= 0 else -0.003
        ax.text(
            val + offset, bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}", va="center",
            ha="left" if val >= 0 else "right",
            fontsize=8, color="#2d3748", fontweight="600",
        )

    pos_patch = mpatches.Patch(color="#e53e3e", alpha=0.9, label="Increases risk")
    neg_patch = mpatches.Patch(color="#38a169", alpha=0.9, label="Decreases risk")
    ax.legend(handles=[pos_patch, neg_patch], fontsize=8, loc="lower right",
              framealpha=0.8, edgecolor="#e2e8f0")

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

with right:
    st.markdown("#### 🎯 Departure Reason Probabilities")
    st.caption("AI model confidence across all departure categories")

    reason_df = pd.DataFrame(result["all_reasons"], columns=["Category", "Probability"])
    reason_df = reason_df[reason_df["Probability"] > 0.005].sort_values("Probability", ascending=False)

    for _, row in reason_df.iterrows():
        cat    = row["Category"]
        p      = row["Probability"]
        r_info = REASON_ACTIONS.get(cat, REASON_ACTIONS["Other"])
        is_top = cat == category
        border = f"2px solid {r_info['color']}" if is_top else "1px solid #e2e8f0"
        bg     = f"{r_info['color']}10" if is_top else "transparent"
        st.markdown(f"""
        <div class="reason-bar-wrap" style="background:{bg}; border:{border};
             border-radius:8px; padding:8px 12px">
            <div class="reason-bar-label">
                <span>{r_info['icon']} <b>{cat}</b>{"  ✓" if is_top else ""}</span>
                <span style="color:{r_info['color']}; font-weight:700">{p:.0%}</span>
            </div>
            <div class="reason-bar-bg">
                <div style="background:{r_info['color']}; width:{int(p*100)}%;
                            height:10px; border-radius:6px; transition:width 0.4s"></div>
            </div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("#### ✅ HR Recommendations")

tab_reason, tab_shap = st.tabs([
    f"{reason_info['icon']} Based on Departure Reason",
    "🔍 Based on SHAP Signals",
])

# Tab 1 — Rule-based reason
with tab_reason:
    st.markdown(f"""
    <div style="background:{reason_info['color']}10; border-left:4px solid {reason_info['color']};
         border-radius:0 8px 8px 0; padding:12px 16px; margin-bottom:16px">
        <b>{reason_info['icon']} {category}</b> — {reason_info['summary']}
    </div>""", unsafe_allow_html=True)
    for i, action in enumerate(reason_info["actions"], 1):
        st.markdown(f"""
        <div class="action-item">
            <span class="action-num">{i}</span>{action}
        </div>""", unsafe_allow_html=True)

# Tab 3 — SHAP-based
with tab_shap:
    for feat in top3_feats:
        val     = float(feature_row.iloc[0][feat])
        shap_v  = float(shap_vals[feat])
        label, action = get_shap_label(feat, shap_v)
        direction     = "▲ high" if shap_v > 0 else "▼ low"
        d_color       = "#e53e3e" if shap_v > 0 else "#38a169"
        with st.expander(f"**{feat}** = {val:.1f}  [{direction}]  →  *{label}*"):
            st.markdown(f"""
            <div class="action-item" style="border-left-color:{d_color}">
                💡 {action}
            </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PLAIN-ENGLISH SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
r1_label, _ = get_shap_label(top3_feats[0], float(shap_vals[top3_feats[0]]))
r2_label, _ = get_shap_label(top3_feats[1], float(shap_vals[top3_feats[1]])) if len(top3_feats) > 1 else ("", "")

st.markdown(f"""
<div class="section-card" style="background:#f7fafc">
    <div style="font-size:0.75rem; font-weight:700; color:#a0aec0;
                letter-spacing:1px; margin-bottom:8px">📝 PLAIN-ENGLISH SUMMARY</div>
    <div style="font-size:1rem; line-height:1.75; color:#2d3748">
        This employee has a <b style="color:{risk_color}">{prob:.0%} predicted probability of leaving</b>
        ({risk_level}).
        The most likely departure reason is <b>{category}</b> (model confidence: {confidence:.0%}).
        The two strongest risk signals are <b>{r1_label.lower()}</b>
        {"and <b>" + r2_label.lower() + "</b>" if r2_label else ""}.
    </div>
</div>""", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; color:#a0aec0; font-size:0.75rem; margin-top:32px;
     padding-top:16px; border-top:1px solid #e2e8f0">
    🔐 No personal identifiers processed · ⚖️ Gender & race removed from model (GDPR Art. 9 + French law)
    · 🇪🇺 EU AI Act — High Risk (Annex III §4) · Decision-support only — human review mandatory
</div>
""", unsafe_allow_html=True)
