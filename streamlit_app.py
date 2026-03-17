"""
HR·AI Retention Advisor — Streamlit Dashboard
Run: C:\\Users\\valen\\anaconda3\\python.exe -m streamlit run streamlit_app.py
"""

import warnings
import pickle
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

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-header {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    padding: 24px 32px; border-radius: 16px; margin-bottom: 24px;
    display: flex; align-items: center; gap: 20px;
}
.logo-svg { flex-shrink: 0; }
.header-text h1 { color: white; font-size: 1.8rem; font-weight: 800; margin: 0; letter-spacing:-0.5px; }
.header-text p  { color: #a0aec0; font-size: 0.88rem; margin: 4px 0 0 0; }
.header-badges  { display:flex; gap:6px; margin-top:8px; flex-wrap:wrap; }
.badge {
    background: rgba(255,255,255,0.08); color: #e2e8f0;
    padding: 3px 10px; border-radius: 20px; font-size: 0.7rem;
    border: 1px solid rgba(255,255,255,0.15);
}
.kpi-card {
    background: white; border: 1px solid #e2e8f0;
    border-radius: 14px; padding: 20px; text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.kpi-value { font-size: 2.4rem; font-weight: 800; line-height: 1; }
.kpi-label { font-size: 0.78rem; color: #718096; margin-top: 6px; }
.kpi-sub   { font-size: 0.9rem; font-weight: 600; margin-top: 4px; }

.section-card {
    background: white; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 20px; margin-bottom: 16px;
}
.action-item {
    background: #f7fafc; border-left: 4px solid #4299e1;
    border-radius: 0 8px 8px 0; padding: 12px 16px;
    margin-bottom: 10px; font-size: 0.88rem;
    line-height: 1.6; color: #2d3748 !important;
}
.action-num {
    display: inline-block; background: #4299e1; color: white;
    border-radius: 50%; width: 22px; height: 22px;
    text-align: center; line-height: 22px;
    font-size: 0.72rem; font-weight: 700; margin-right: 8px;
}
.insight-box {
    background: linear-gradient(135deg, #ebf8ff, #e6fffa);
    border-left: 4px solid #3182ce; border-radius: 0 10px 10px 0;
    padding: 14px 18px; margin-bottom: 10px; color: #2d3748;
    font-size: 0.9rem; line-height: 1.6;
}
.emp-profile-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 8px 0; border-bottom: 1px solid #f0f4f8;
    font-size: 0.88rem;
}
.emp-feat-name  { color: #718096; font-weight: 500; }
.emp-feat-value { color: #2d3748; font-weight: 700; }
.reason-bar-wrap { margin-bottom: 8px; }
.reason-bar-label {
    display: flex; justify-content: space-between;
    font-size: 0.82rem; margin-bottom: 3px;
}
.reason-bar-bg {
    background: #edf2f7; border-radius: 6px; height: 9px; overflow: hidden;
}
div[data-testid="stSidebar"] { background: #f8fafc; border-right: 1px solid #e2e8f0; }
[data-testid="stButton"] button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    padding: 10px !important; box-shadow: 0 4px 12px rgba(102,126,234,0.35) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ─────────────────────────────────────────────────────────────
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
REASON_FEATURE_COLS = art["reason_feature_cols"]
X_ALL               = art.get("X_imputed", pd.DataFrame())
Y_ALL               = art.get("y", pd.Series(dtype=int))

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
        "summary": "Likely leaving due to workload, hours, or work-life balance.",
        "actions": [
            "Review workload distribution within the team.",
            "Offer flexible/hybrid schedule or reduced hours temporarily.",
            "Explore a temporary leave of absence instead of resignation.",
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
            "Explore medical leave, flexible hours, or reduced duties.",
            "Ensure manager is aware of reasonable adjustment obligations.",
        ],
    },
    "Retirement": {
        "icon": "🎓", "color": "#27ae60",
        "summary": "Approaching retirement age.",
        "actions": [
            "Begin knowledge transfer planning — document critical expertise.",
            "Explore phased retirement or part-time consultancy.",
            "Identify and develop an internal successor.",
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
    "absences":             ("high", "High absenteeism",       "Schedule wellness check-in; consider flexible arrangements."),
    "empsatisfaction":      ("low",  "Low satisfaction score", "Arrange 1-on-1; review workload, team dynamics, role fit."),
    "engagementsurvey":     ("low",  "Low engagement",         "Offer stretch projects or mentoring opportunities."),
    "dayslatelast30":       ("high", "Frequent late arrivals", "Discuss scheduling or commute; consider flexible start times."),
    "salary":               ("low",  "Below-market salary",    "Conduct pay equity review; fast-track compensation adjustment."),
    "specialprojectscount": ("low",  "No special projects",    "Assign visible project; connect to mentor outside direct team."),
    "perfscoreid":          ("low",  "Low performance score",  "Set up development plan with clear milestones and coaching."),
    "marriedid":            ("high", "Personal life change",   "Offer EAP resources; review workload and travel requirements."),
    "recruitmentsource":    ("high", "Recruitment channel",    "Review onboarding quality; strengthen internal mobility path."),
}

def get_shap_label(feature: str, shap_val: float):
    key = feature.lower()
    direction = "high" if shap_val > 0 else "low"
    for feat_key, (dir_key, reason, action) in SHAP_LABELS.items():
        if feat_key in key and dir_key == direction:
            return reason, action
    dir_word = "high" if shap_val > 0 else "low"
    return f"{dir_word} {feature}", "Review this metric with the line manager."

# ── Core prediction ────────────────────────────────────────────────────────────
def build_row(inputs: dict) -> pd.DataFrame:
    row = {col: MEDIANS.get(col, 0.0) for col in FEATURE_COLS}
    for k, v in inputs.items():
        if k in row:
            row[k] = float(v)
    return pd.DataFrame([row])[FEATURE_COLS]

def predict(feature_row: pd.DataFrame) -> dict:
    prob = float(rf_model.predict_proba(feature_row)[0, 1])
    sv   = explainer.shap_values(feature_row)
    if isinstance(sv, list):   sv_pos = sv[1][0]
    elif sv.ndim == 3:         sv_pos = sv[0, :, 1]
    else:                      sv_pos = sv[0]
    shap_s   = pd.Series(sv_pos, index=FEATURE_COLS)
    feat_r   = feature_row.iloc[0].reindex(REASON_FEATURE_COLS, fill_value=0)
    r_proba  = reason_model.predict_proba(feat_r.values.reshape(1, -1))[0]
    top_idx  = int(np.argmax(r_proba))
    return {
        "prob":       prob,
        "shap":       shap_s,
        "category":   reason_model.classes_[top_idx],
        "confidence": float(r_proba[top_idx]),
        "all_reasons": sorted(zip(reason_model.classes_, r_proba), key=lambda x: -x[1]),
    }

@st.cache_data
def batch_predict(_X: pd.DataFrame) -> np.ndarray:
    return rf_model.predict_proba(_X)[:, 1]

# ── Risk helpers ───────────────────────────────────────────────────────────────
def risk_meta(prob):
    if prob >= 0.70: return "HIGH RISK",   "#e53e3e", "🔴"
    if prob >= 0.40: return "MEDIUM RISK", "#dd6b20", "🟠"
    return "LOW RISK", "#38a169", "🟢"

# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
  <svg class="logo-svg" width="64" height="64" viewBox="0 0 64 64">
    <defs>
      <linearGradient id="lg1" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" style="stop-color:#e94560"/>
        <stop offset="100%" style="stop-color:#764ba2"/>
      </linearGradient>
    </defs>
    <circle cx="32" cy="32" r="30" fill="url(#lg1)" opacity="0.9"/>
    <text x="32" y="38" text-anchor="middle" font-size="26" fill="white">🧠</text>
  </svg>
  <div class="header-text">
    <h1>HR · AI Retention Advisor</h1>
    <p>Predict · Explain · Act — Responsible AI for HR Decision Support</p>
    <div class="header-badges">
      <span class="badge">🔐 GDPR Compliant</span>
      <span class="badge">⚖️ No Demographic Bias</span>
      <span class="badge">🔍 SHAP Explainable</span>
      <span class="badge">🌿 Frugal AI</span>
      <span class="badge">🇪🇺 EU AI Act — High Risk</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — Navigation
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧭 Navigation")
    page = st.radio(
        "Navigation", ["📊 Global Dashboard", "👤 Lookup by Employee ID", "🔍 Manual Analysis"],
        label_visibility="collapsed",
    )
    st.divider()

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — GLOBAL DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Global Dashboard":

    if X_ALL.empty:
        st.warning("Full dataset not found in artifacts. Re-run the **Save Model Artifacts** cell.")
        st.stop()

    probas = batch_predict(X_ALL)
    n      = len(probas)
    n_high = int((probas >= 0.70).sum())
    n_med  = int(((probas >= 0.40) & (probas < 0.70)).sum())
    n_low  = int((probas < 0.40).sum())
    actual_rate = float(Y_ALL.mean()) if len(Y_ALL) else 0.0

    # ── KPI row ────────────────────────────────────────────────────────────────
    st.markdown("### 📊 Company-Wide Turnover Risk Overview")
    k1, k2, k3, k4, k5 = st.columns(5)
    for col, value, label, sub, color in [
        (k1, n,               "Total Employees",     "in dataset",           "#4299e1"),
        (k2, f"{actual_rate:.0%}", "Actual Turnover",    "historical rate",     "#e53e3e"),
        (k3, n_high,          "🔴 High Risk",         "prob ≥ 70%",           "#e53e3e"),
        (k4, n_med,           "🟠 Medium Risk",       "prob 40–70%",          "#dd6b20"),
        (k5, n_low,           "🟢 Low Risk",          "prob < 40%",           "#38a169"),
    ]:
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="border-top:4px solid {color}">
                <div class="kpi-value" style="color:{color}">{value}</div>
                <div class="kpi-sub">{label}</div>
                <div class="kpi-label">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts row ─────────────────────────────────────────────────────────────
    ch1, ch2 = st.columns(2)

    with ch1:
        st.markdown("#### Risk Score Distribution")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("#fafafa")
        n_bins, bins, patches = ax.hist(probas, bins=20, edgecolor="white", linewidth=0.6)
        for patch, left in zip(patches, bins[:-1]):
            if left >= 0.70:   patch.set_facecolor("#e53e3e")
            elif left >= 0.40: patch.set_facecolor("#dd6b20")
            else:              patch.set_facecolor("#38a169")
        ax.axvline(0.40, color="#dd6b20", linestyle="--", linewidth=1.5, label="Medium (40%)")
        ax.axvline(0.70, color="#e53e3e", linestyle="--", linewidth=1.5, label="High (70%)")
        ax.set_xlabel("Predicted Turnover Probability", fontsize=9, color="#4a5568")
        ax.set_ylabel("Employees", fontsize=9, color="#4a5568")
        ax.set_title("Distribution of Risk Scores", fontsize=11, fontweight="bold", color="#2d3748")
        ax.spines[["top","right"]].set_visible(False)
        ax.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with ch2:
        st.markdown("#### Departure Reason Forecast")
        # predict reason for all at-risk employees
        at_risk_idx = [i for i, p in enumerate(probas) if p >= 0.40]
        reason_counts = {}
        for i in at_risk_idx:
            row  = X_ALL.iloc[[i]]
            feat = row.iloc[0].reindex(REASON_FEATURE_COLS, fill_value=0)
            r_p  = reason_model.predict_proba(feat.values.reshape(1, -1))[0]
            cat  = reason_model.classes_[int(np.argmax(r_p))]
            reason_counts[cat] = reason_counts.get(cat, 0) + 1

        if reason_counts:
            rc_s = pd.Series(reason_counts).sort_values(ascending=True)
            colors_r = [REASON_ACTIONS.get(c, REASON_ACTIONS["Other"])["color"] for c in rc_s.index]
            icons_r  = [REASON_ACTIONS.get(c, REASON_ACTIONS["Other"])["icon"]  for c in rc_s.index]
            fig2, ax2 = plt.subplots(figsize=(6, 3.5))
            fig2.patch.set_facecolor("white")
            ax2.set_facecolor("#fafafa")
            bars2 = ax2.barh([f"{ic} {c}" for ic, c in zip(icons_r, rc_s.index)],
                             rc_s.values, color=colors_r, edgecolor="white", alpha=0.85)
            ax2.set_title("Predicted Departure Reasons\n(at-risk employees)", fontsize=11,
                          fontweight="bold", color="#2d3748")
            ax2.set_xlabel("Number of employees", fontsize=9, color="#4a5568")
            ax2.spines[["top","right"]].set_visible(False)
            for bar, v in zip(bars2, rc_s.values):
                ax2.text(v + 0.1, bar.get_y() + bar.get_height()/2,
                         str(v), va="center", fontsize=8, fontweight="700")
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Top 10 at-risk employees ───────────────────────────────────────────────
    st.markdown("#### 🔴 Top 10 Highest-Risk Employees")
    top10_idx = np.argsort(probas)[::-1][:10]
    rows = []
    for rank, i in enumerate(top10_idx, 1):
        p    = probas[i]
        rl, rc, re = risk_meta(p)
        feat = X_ALL.iloc[[i]].iloc[0].reindex(REASON_FEATURE_COLS, fill_value=0)
        r_p  = reason_model.predict_proba(feat.values.reshape(1,-1))[0]
        cat  = reason_model.classes_[int(np.argmax(r_p))]
        rows.append({
            "Rank":        rank,
            "Employee #":  i,
            "Risk Score":  f"{p:.0%}",
            "Level":       f"{re} {rl}",
            "Predicted Reason": f"{REASON_ACTIONS.get(cat,REASON_ACTIONS['Other'])['icon']} {cat}",
            "Salary":      f"${X_ALL.iloc[i].get('Salary', 0):,.0f}" if 'Salary' in X_ALL.columns else "—",
            "Absences":    int(X_ALL.iloc[i].get('Absences', 0)) if 'Absences' in X_ALL.columns else "—",
            "Satisfaction":int(X_ALL.iloc[i].get('EmpSatisfaction', 0)) if 'EmpSatisfaction' in X_ALL.columns else "—",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Rank"), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Key takeaways ──────────────────────────────────────────────────────────
    st.markdown("#### 💡 Key Takeaways for HR Leadership")
    top_reason = max(reason_counts, key=reason_counts.get) if reason_counts else "Unknown"
    pct_at_risk = (n_high + n_med) / n
    top_reason_info = REASON_ACTIONS.get(top_reason, REASON_ACTIONS["Other"])

    insights = [
        f"**{pct_at_risk:.0%} of your workforce** ({n_high + n_med} employees) are predicted at medium or high risk of leaving. Immediate retention conversations are recommended for the {n_high} high-risk employees.",
        f"The most frequent predicted departure reason is **{top_reason_info['icon']} {top_reason}** ({reason_counts.get(top_reason,0)} employees). {top_reason_info['actions'][0]}",
        f"**Absences and engagement scores** are the strongest systemic signals in the model. Running quarterly engagement surveys and monitoring absence trends early can reduce turnover by up to 25%.",
        f"**{n_low} employees ({n_low/n:.0%})** are low risk. Focus immediate HR bandwidth on the {n_high} high-risk cases — the model gives you a priority list.",
        "This model does not use gender or race. Fairness is ensured structurally: no demographic attribute was ever seen during training (GDPR Art. 9 + French law).",
    ]
    for txt in insights:
        st.markdown(f'<div class="insight-box">💡 {txt}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — LOOKUP BY EMPLOYEE ID
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👤 Lookup by Employee ID":

    if X_ALL.empty:
        st.warning("Full dataset not in artifacts. Re-run the **Save Model Artifacts** cell.")
        st.stop()

    with st.sidebar:
        st.markdown("### 👤 Select Employee")
        emp_id = st.number_input(
            "Employee Index (0 – " + str(len(X_ALL)-1) + ")",
            min_value=0, max_value=len(X_ALL)-1, value=0, step=1,
        )
        analyse_btn = st.button("🔍 Analyse Employee", use_container_width=True)

    if not analyse_btn:
        st.info("Select an **Employee Index** in the sidebar and click **Analyse Employee**.")
        st.stop()

    feat_row = X_ALL.iloc[[emp_id]]
    result   = predict(feat_row)
    prob, shap_vals, category, confidence = (
        result["prob"], result["shap"], result["category"], result["confidence"]
    )
    risk_level, risk_color, risk_emoji = risk_meta(prob)
    reason_info = REASON_ACTIONS.get(category, REASON_ACTIONS["Other"])
    top3_feats  = shap_vals.abs().sort_values(ascending=False).head(3).index.tolist()

    st.markdown(f"### {risk_emoji} Employee #{emp_id} — Risk Report")

    # KPI row
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="kpi-card" style="border-top:4px solid {risk_color}">
            <div class="kpi-value" style="color:{risk_color}">{prob:.0%}</div>
            <div class="kpi-sub">{risk_emoji} {risk_level}</div>
            <div class="kpi-label">Predicted turnover probability</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        cat_color = reason_info["color"]
        st.markdown(f"""
        <div class="kpi-card" style="border-top:4px solid {cat_color}">
            <div class="kpi-value">{reason_info['icon']}</div>
            <div class="kpi-sub" style="color:{cat_color}; font-size:0.85rem">{category}</div>
            <div class="kpi-label">Predicted departure reason · {confidence:.0%} confidence</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        top1_label, _ = get_shap_label(top3_feats[0], float(shap_vals[top3_feats[0]]))
        st.markdown(f"""
        <div class="kpi-card" style="border-top:4px solid #4299e1">
            <div class="kpi-value" style="color:#2b6cb0; font-size:1.3rem">{top3_feats[0]}</div>
            <div class="kpi-sub" style="color:#2b6cb0; font-size:0.82rem">{top1_label}</div>
            <div class="kpi-label">Strongest SHAP signal</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Profile + SHAP side by side
    left, right = st.columns([1, 1.4])

    with left:
        st.markdown("#### 📋 Employee Profile")
        display_cols = [c for c in [
            "Salary","EmpSatisfaction","EngagementSurvey","Absences",
            "DaysLateLast30","SpecialProjectsCount","PerfScoreID","MarriedID"
        ] if c in feat_row.columns]
        for col in display_cols:
            val = feat_row.iloc[0][col]
            st.markdown(f"""
            <div class="emp-profile-row">
                <span class="emp-feat-name">{col}</span>
                <span class="emp-feat-value">{val:.1f}</span>
            </div>""", unsafe_allow_html=True)

    with right:
        st.markdown("#### 📊 SHAP Risk Factors")
        top_n    = shap_vals.abs().sort_values(ascending=False).head(8)
        top_shap = shap_vals[top_n.index].sort_values()
        fig, ax  = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("#fafafa")
        colors = ["#e53e3e" if v > 0 else "#38a169" for v in top_shap.values]
        bars   = ax.barh([f.replace("_"," ") for f in top_shap.index],
                          top_shap.values, color=colors, edgecolor="white", height=0.6, alpha=0.9)
        ax.axvline(0, color="#2d3748", linewidth=1.2)
        ax.set_xlabel("SHAP value", fontsize=9, color="#4a5568")
        ax.spines[["top","right"]].set_visible(False)
        for bar, val in zip(bars, top_shap.values):
            offset = 0.003 if val >= 0 else -0.003
            ax.text(val+offset, bar.get_y()+bar.get_height()/2,
                    f"{val:+.3f}", va="center",
                    ha="left" if val>=0 else "right",
                    fontsize=8, color="#2d3748", fontweight="600")
        pos_p = mpatches.Patch(color="#e53e3e", alpha=0.9, label="Increases risk")
        neg_p = mpatches.Patch(color="#38a169", alpha=0.9, label="Decreases risk")
        ax.legend(handles=[pos_p, neg_p], fontsize=8, framealpha=0.8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Recommendations
    st.markdown("#### ✅ HR Recommendations")
    tab_r, tab_s = st.tabs([f"{reason_info['icon']} Departure Reason", "🔍 SHAP Signals"])
    with tab_r:
        st.markdown(f"""
        <div style="background:{reason_info['color']}10; border-left:4px solid {reason_info['color']};
             border-radius:0 8px 8px 0; padding:12px 16px; margin-bottom:12px; color:#2d3748">
            <b>{reason_info['icon']} {category}</b> — {reason_info['summary']}
        </div>""", unsafe_allow_html=True)
        for i, action in enumerate(reason_info["actions"], 1):
            st.markdown(f"""
            <div class="action-item">
                <span class="action-num">{i}</span>{action}
            </div>""", unsafe_allow_html=True)
    with tab_s:
        for feat in top3_feats:
            val    = float(feat_row.iloc[0][feat])
            sv     = float(shap_vals[feat])
            label, action = get_shap_label(feat, sv)
            direction     = "▲ high" if sv > 0 else "▼ low"
            d_color       = "#e53e3e" if sv > 0 else "#38a169"
            with st.expander(f"**{feat}** = {val:.1f}  [{direction}]  →  *{label}*"):
                st.markdown(f"""
                <div class="action-item" style="border-left-color:{d_color}; color:#2d3748">
                    💡 {action}
                </div>""", unsafe_allow_html=True)

    # Summary sentence
    r1, _ = get_shap_label(top3_feats[0], float(shap_vals[top3_feats[0]]))
    r2, _ = get_shap_label(top3_feats[1], float(shap_vals[top3_feats[1]])) if len(top3_feats)>1 else ("","")
    st.markdown(f"""
    <div class="section-card" style="background:#f7fafc; margin-top:12px">
        <div style="font-size:0.72rem; font-weight:700; color:#a0aec0; letter-spacing:1px; margin-bottom:8px">
            📝 PLAIN-ENGLISH SUMMARY
        </div>
        <div style="font-size:0.95rem; line-height:1.75; color:#2d3748">
            Employee #{emp_id} has a <b style="color:{risk_color}">{prob:.0%} predicted probability of leaving</b>
            ({risk_level}). Most likely reason: <b>{category}</b> ({confidence:.0%} confidence).
            Key signals: <b>{r1.lower()}</b>{"  and  <b>" + r2.lower() + "</b>" if r2 else ""}.
        </div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — MANUAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Manual Analysis":

    with st.sidebar:
        st.markdown("### 💼 Compensation & Role")
        salary = st.number_input(
            "Annual Salary (USD)", min_value=30_000, max_value=300_000,
            value=int(MEDIANS.get("Salary", 60_000)), step=1_000,
        )
        perf_label = st.selectbox(
            "Performance Score",
            ["1 — PIP (serious issues)", "2 — Needs Improvement",
             "3 — Fully Meets Expectations", "4 — Exceeds Expectations"], index=2,
        )
        perf_score       = int(perf_label[0])
        special_projects = st.slider("Special Projects Count", 0, 10,
                                     value=int(MEDIANS.get("SpecialProjectsCount", 1)))

        st.markdown("### 📊 Engagement")
        satisfaction = st.slider("Employee Satisfaction  (1–5)", 1, 5,
                                  value=int(MEDIANS.get("EmpSatisfaction", 3)))
        engagement   = st.slider("Engagement Survey  (1.0–5.0)", 1.0, 5.0,
                                  value=round(float(MEDIANS.get("EngagementSurvey", 3.5)), 1), step=0.1)

        st.markdown("### 🚨 Behavioural Signals")
        absences  = st.number_input("Absences (days/year)", min_value=0, max_value=100,
                                     value=int(MEDIANS.get("Absences", 8)))
        days_late = st.number_input("Days Late (last 30 days)", min_value=0, max_value=30,
                                     value=int(MEDIANS.get("DaysLateLast30", 0)))

        st.divider()
        analyse = st.button("🔍  Analyse Employee", use_container_width=True, type="primary")

    if not analyse:
        c1, c2, c3 = st.columns(3)
        for col, icon, title, desc in [
            (c1, "🎯", "Predict",  "Assess turnover risk with a Random Forest model trained on HR data"),
            (c2, "🔍", "Explain",  "SHAP waterfall chart shows exactly which factors drive the prediction"),
            (c3, "💡", "Act",      "Personalised retention plan based on departure reason and SHAP signals"),
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
            <b>⚠️ Decision-support only.</b> All predictions must be reviewed by a qualified HR
            professional before any action is taken. No personal identifiers are processed.
            EU AI Act classification: <b>HIGH RISK</b> (Annex III §4).
        </div>""", unsafe_allow_html=True)
        st.stop()

    user_inputs = {
        "Salary": salary, "EmpSatisfaction": satisfaction,
        "EngagementSurvey": engagement, "Absences": absences,
        "DaysLateLast30": days_late, "SpecialProjectsCount": special_projects,
        "PerfScoreID": perf_score,
    }
    feature_row = build_row(user_inputs)
    result      = predict(feature_row)
    prob, shap_vals, category, confidence = (
        result["prob"], result["shap"], result["category"], result["confidence"]
    )
    risk_level, risk_color, risk_emoji = risk_meta(prob)
    reason_info = REASON_ACTIONS.get(category, REASON_ACTIONS["Other"])
    top3_feats  = shap_vals.abs().sort_values(ascending=False).head(3).index.tolist()

    # KPI cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="kpi-card" style="border-top:4px solid {risk_color}">
            <div class="kpi-value" style="color:{risk_color}">{prob:.0%}</div>
            <div class="kpi-sub">{risk_emoji} {risk_level}</div>
            <div class="kpi-label">Predicted turnover probability</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        cat_color = reason_info["color"]
        st.markdown(f"""
        <div class="kpi-card" style="border-top:4px solid {cat_color}">
            <div class="kpi-value">{reason_info['icon']}</div>
            <div class="kpi-sub" style="color:{cat_color}; font-size:0.85rem">{category}</div>
            <div class="kpi-label">Predicted departure reason · {confidence:.0%} confidence</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        top1_label, _ = get_shap_label(top3_feats[0], float(shap_vals[top3_feats[0]]))
        st.markdown(f"""
        <div class="kpi-card" style="border-top:4px solid #4299e1">
            <div class="kpi-value" style="color:#2b6cb0; font-size:1.3rem">{top3_feats[0]}</div>
            <div class="kpi-sub" style="color:#2b6cb0; font-size:0.82rem">{top1_label}</div>
            <div class="kpi-label">Strongest SHAP signal</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # SHAP + Reason breakdown
    left, right = st.columns([1.3, 1])
    with left:
        st.markdown("#### 📊 SHAP Risk Factor Analysis")
        st.caption("🔴 Red = increases risk · 🟢 Green = decreases risk")
        top_n    = shap_vals.abs().sort_values(ascending=False).head(8)
        top_shap = shap_vals[top_n.index].sort_values()
        fig, ax  = plt.subplots(figsize=(7, 4.5))
        fig.patch.set_facecolor("white"); ax.set_facecolor("#fafafa")
        colors = ["#e53e3e" if v > 0 else "#38a169" for v in top_shap.values]
        bars   = ax.barh([f.replace("_"," ") for f in top_shap.index],
                          top_shap.values, color=colors, edgecolor="white", height=0.65, alpha=0.9)
        ax.axvline(0, color="#2d3748", linewidth=1.2, zorder=5)
        ax.set_xlabel("SHAP value", fontsize=9, color="#4a5568")
        ax.set_title("Feature Contributions to Turnover Risk", fontsize=11,
                     fontweight="bold", color="#2d3748")
        ax.spines[["top","right"]].set_visible(False)
        for bar, val in zip(bars, top_shap.values):
            offset = 0.003 if val >= 0 else -0.003
            ax.text(val+offset, bar.get_y()+bar.get_height()/2, f"{val:+.3f}",
                    va="center", ha="left" if val>=0 else "right",
                    fontsize=8, color="#2d3748", fontweight="600")
        pos_p = mpatches.Patch(color="#e53e3e", alpha=0.9, label="Increases risk")
        neg_p = mpatches.Patch(color="#38a169", alpha=0.9, label="Decreases risk")
        ax.legend(handles=[pos_p, neg_p], fontsize=8, framealpha=0.8, edgecolor="#e2e8f0")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    with right:
        st.markdown("#### 🎯 Departure Reason Probabilities")
        reason_df = pd.DataFrame(result["all_reasons"], columns=["Category","Probability"])
        reason_df = reason_df[reason_df["Probability"] > 0.005].sort_values("Probability", ascending=False)
        for _, row in reason_df.iterrows():
            cat    = row["Category"]; p = row["Probability"]
            r_info = REASON_ACTIONS.get(cat, REASON_ACTIONS["Other"])
            is_top = cat == category
            border = f"2px solid {r_info['color']}" if is_top else "1px solid #e2e8f0"
            bg     = f"{r_info['color']}10"         if is_top else "transparent"
            st.markdown(f"""
            <div class="reason-bar-wrap" style="background:{bg}; border:{border};
                 border-radius:8px; padding:8px 12px">
                <div class="reason-bar-label">
                    <span>{r_info['icon']} <b>{cat}</b>{"  ✓" if is_top else ""}</span>
                    <span style="color:{r_info['color']}; font-weight:700">{p:.0%}</span>
                </div>
                <div class="reason-bar-bg">
                    <div style="background:{r_info['color']}; width:{int(p*100)}%;
                                height:9px; border-radius:6px"></div>
                </div>
            </div>""", unsafe_allow_html=True)

    # Recommendations
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### ✅ HR Recommendations")
    tab_r, tab_s = st.tabs([f"{reason_info['icon']} Departure Reason", "🔍 SHAP Signals"])
    with tab_r:
        st.markdown(f"""
        <div style="background:{reason_info['color']}10; border-left:4px solid {reason_info['color']};
             border-radius:0 8px 8px 0; padding:12px 16px; margin-bottom:12px; color:#2d3748">
            <b>{reason_info['icon']} {category}</b> — {reason_info['summary']}
        </div>""", unsafe_allow_html=True)
        for i, action in enumerate(reason_info["actions"], 1):
            st.markdown(f"""
            <div class="action-item">
                <span class="action-num">{i}</span>{action}
            </div>""", unsafe_allow_html=True)
    with tab_s:
        for feat in top3_feats:
            val    = float(feature_row.iloc[0][feat])
            sv     = float(shap_vals[feat])
            label, action = get_shap_label(feat, sv)
            direction     = "▲ high" if sv > 0 else "▼ low"
            d_color       = "#e53e3e" if sv > 0 else "#38a169"
            with st.expander(f"**{feat}** = {val:.1f}  [{direction}]  →  *{label}*"):
                st.markdown(f"""
                <div class="action-item" style="border-left-color:{d_color}; color:#2d3748">
                    💡 {action}
                </div>""", unsafe_allow_html=True)

    # Summary
    r1, _ = get_shap_label(top3_feats[0], float(shap_vals[top3_feats[0]]))
    r2, _ = get_shap_label(top3_feats[1], float(shap_vals[top3_feats[1]])) if len(top3_feats)>1 else ("","")
    st.markdown(f"""
    <div class="section-card" style="background:#f7fafc">
        <div style="font-size:0.72rem; font-weight:700; color:#a0aec0; letter-spacing:1px; margin-bottom:8px">
            📝 PLAIN-ENGLISH SUMMARY
        </div>
        <div style="font-size:0.95rem; line-height:1.75; color:#2d3748">
            This employee has a <b style="color:{risk_color}">{prob:.0%} predicted probability of leaving</b>
            ({risk_level}). Most likely reason: <b>{category}</b> ({confidence:.0%} confidence).
            Key signals: <b>{r1.lower()}</b>{"  and  <b>" + r2.lower() + "</b>" if r2 else ""}.
        </div>
    </div>""", unsafe_allow_html=True)

# ── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; color:#a0aec0; font-size:0.72rem; margin-top:32px;
     padding-top:16px; border-top:1px solid #e2e8f0">
    🔐 No personal identifiers processed · ⚖️ Gender & race removed (GDPR Art. 9 + French law)
    · 🇪🇺 EU AI Act — High Risk (Annex III §4) · Decision-support only — human review mandatory
</div>
""", unsafe_allow_html=True)
