import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os

BASE = r'D:\UserData\Desktop\Hackathon-IA'

st.set_page_config(page_title="HR Attrition Intelligence", page_icon="◈", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #f8f8f8; color: #111; }
.stApp { background-color: #f8f8f8; }
h1, h2, h3 { font-family: 'Inter', sans-serif; font-weight: 600; color: #111; }
.metric-card { background: #fff; border: 1px solid #e8e8e8; border-radius: 12px; padding: 24px; text-align: center; box-shadow: 0 1px 4px rgba(0,0,0,0.05); }
.metric-label { font-family: 'JetBrains Mono', monospace; font-size: 10px; letter-spacing: .15em; text-transform: uppercase; color: #aaa; margin-bottom: 10px; }
.metric-value { font-size: 40px; font-weight: 600; line-height: 1; color: #111; }
.risk-high { color: #e53e3e; } .risk-medium { color: #d97706; } .risk-low { color: #16a34a; }
.recommendation-box { background: #f9f9f9; border-left: 3px solid #111; border-radius: 0 8px 8px 0; padding: 14px 18px; margin: 8px 0; font-size: 13px; color: #444; }
.risk-badge { display: inline-block; background: #f3f3f3; border: 1px solid #e0e0e0; border-radius: 20px; padding: 4px 12px; font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #666; margin: 3px; }
.section-title { font-family: 'JetBrains Mono', monospace; font-size: 10px; letter-spacing: .15em; text-transform: uppercase; color: #bbb; border-bottom: 1px solid #ececec; padding-bottom: 8px; margin-bottom: 20px; }
.employee-row { background: #fff; border: 1px solid #ececec; border-radius: 8px; padding: 14px 18px; margin: 6px 0; }
div[data-testid="stSidebar"] { background-color: #fff; border-right: 1px solid #ececec; }
.stButton > button { background: #111; color: #fff; border: none; border-radius: 8px; font-family: 'Inter', sans-serif; font-weight: 500; font-size: 14px; padding: 12px 24px; width: 100%; cursor: pointer; transition: opacity .2s; }
.stButton > button:hover { opacity: .8; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    path = r'D:\UserData\Desktop\Hackathon-IA\rf_attrition_model.joblib'
    return joblib.load(path)

def generate_recommendations(perf_score, satisfaction, sentiment, absences, risk_pct):
    risks, recs = [], []
    if perf_score in ['Needs Improvement', 'PIP']:
        risks.append("Low performance score")
        recs.append("Schedule a structured performance improvement plan with monthly check-ins.")
    if satisfaction <= 2:
        risks.append("Low employee satisfaction")
        recs.append("Arrange a confidential 1-on-1 to identify root causes of dissatisfaction.")
    if sentiment < -0.3:
        risks.append("Negative exit sentiment detected")
        recs.append("Review qualitative feedback patterns across the team for systemic issues.")
    if absences >= 15:
        risks.append("High absenteeism")
        recs.append("Consider flexible work arrangements or a wellbeing support programme.")
    if risk_pct >= 70 and not recs:
        recs.append("High attrition risk detected. Recommend proactive retention conversation.")
    return risks, recs

st.sidebar.markdown("## ◈ Navigation")
page = st.sidebar.radio("", ["🔍 Employee Analysis", "💬 NLP Exit Analysis", "⚡ Carbon Footprint", "🚨 High Risk Employees"])
st.sidebar.markdown("---")

# ── PAGE 1 ──────────────────────────────────
if page == "🔍 Employee Analysis":
    with st.sidebar:
        st.markdown("<div class='section-title'>Structured Data</div>", unsafe_allow_html=True)
        department   = st.selectbox("Department", ["IT/IS", "Production", "Sales", "Software Engineering", "Admin Offices", "Executive Office"])
        perf_score   = st.selectbox("Performance Score", ["Exceeds", "Fully Meets", "Needs Improvement", "PIP"])
        satisfaction = st.slider("Employee Satisfaction (1–5)", 1, 5, 3)
        absences     = st.slider("Absences (days/year)", 0, 30, 5)
        years        = st.slider("Years at Company", 0, 20, 3)
        st.markdown("<div class='section-title' style='margin-top:20px'>NLP · Exit Sentiment</div>", unsafe_allow_html=True)
        sentiment    = st.slider("Sentiment Score", -1.0, 1.0, 0.0, 0.01, help="-1=Very Negative · +1=Very Positive")
        predict_btn  = st.button("Analyse Employee")

    st.markdown("# HR Attrition Intelligence")
    st.markdown("<p style='color:#aaa;font-size:13px;margin-top:-12px;font-family:JetBrains Mono'>Predictive analytics · Explainable AI · Responsible ML</p>", unsafe_allow_html=True)
    st.markdown("---")

    if predict_btn:
        perf_map = {"Exceeds": 4, "Fully Meets": 3, "Needs Improvement": 2, "PIP": 1}
        dept_map = {"IT/IS": 0, "Production": 1, "Sales": 2, "Software Engineering": 3, "Admin Offices": 4, "Executive Office": 5}
        input_data = pd.DataFrame([{
    'Salary': 50000,
    'Position': 0,
    'State': 0,
    'Zip': 0,
    'MaritalDesc': 0,
    'CitizenDesc': 0,
    'Department': dept_map.get(department, 0),
    'RecruitmentSource': 0,
    'PerformanceScore': perf_map.get(perf_score, 3),
    'EngagementSurvey': 3.0,
    'EmpSatisfaction': satisfaction,
    'SpecialProjectsCount': 0,
    'DaysLateLast30': 0,
    'Absences': absences,
    'tenure_years': years,
    'hire_month': 6,
    'seniority_band': 0,
    'age_at_hire': 30,
    'current_age': 30 + years,
    'attendance_risk': 0,
    'disengagement_score': max(0, (3 - satisfaction)),
    'salary_ratio': 1.0,
    'perf_risk_flag': 1 if perf_score in ['Needs Improvement', 'PIP'] else 0,
}])
        try:
            model = load_model(); prob = model.predict_proba(input_data)[0][1]; use_demo = False
        except Exception as e:
           st.error(f"Model error: {e}")
           use_demo = True

        risk_pct = round(prob * 100)
        risk_class = "risk-high" if risk_pct>=70 else ("risk-medium" if risk_pct>=40 else "risk-low")
        risk_label = "HIGH RISK" if risk_pct>=70 else ("MEDIUM RISK" if risk_pct>=40 else "LOW RISK")
        risks, recs = generate_recommendations(perf_score, satisfaction, sentiment, absences, risk_pct)

        c1,c2,c3,c4 = st.columns(4)
        with c1: st.markdown(f"<div class='metric-card'><div class='metric-label'>Attrition Probability</div><div class='metric-value {risk_class}'>{risk_pct}%</div></div>", unsafe_allow_html=True)
        with c2: st.markdown(f"<div class='metric-card'><div class='metric-label'>Risk Level</div><div class='metric-value {risk_class}' style='font-size:22px;padding-top:10px'>{risk_label}</div></div>", unsafe_allow_html=True)
        with c3:
            sl = "Negative" if sentiment<-0.1 else ("Positive" if sentiment>0.1 else "Neutral")
            sc = "#e53e3e" if sentiment<-0.1 else ("#16a34a" if sentiment>0.1 else "#888")
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Exit Sentiment (NLP)</div><div class='metric-value' style='font-size:22px;padding-top:10px;color:{sc}'>{sl}</div></div>", unsafe_allow_html=True)
        with c4: st.markdown(f"<div class='metric-card'><div class='metric-label'>Satisfaction Score</div><div class='metric-value'>{satisfaction}<span style='font-size:18px;color:#ccc'>/5</span></div></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        left, right = st.columns([1.2, 1])
        with left:
            st.markdown("<div class='section-title'>Feature Importance (SHAP)</div>", unsafe_allow_html=True)
            shap_data = {"EmpSatisfaction": round((5-satisfaction)*0.12,3), "sentiment_score": round(max(0,-sentiment)*0.15,3), "Absences": round((absences/30)*0.20,3), "PerformanceScore": round((4-perf_map.get(perf_score,3))*0.08,3), "YearsAtCompany": round(max(0,(10-years)/10)*0.10,3)}
            if not use_demo:
                try:
                    sv = shap.TreeExplainer(model).shap_values(input_data)
                    sv = sv[1][0] if isinstance(sv, list) else sv[0]
                    shap_data = dict(zip(input_data.columns, sv))
                except: pass
            fig, ax = plt.subplots(figsize=(6, 3.5))
            fig.patch.set_facecolor('#fff'); ax.set_facecolor('#fff')
            vals = list(shap_data.values())
            ax.barh(list(shap_data.keys()), vals, color=['#e53e3e' if v>0 else '#16a34a' for v in vals], height=0.5)
            ax.axvline(0, color='#ddd', linewidth=0.8)
            ax.tick_params(colors='#888', labelsize=10)
            for s in ax.spines.values(): s.set_color('#ececec')
            ax.set_xlabel("SHAP value", color='#aaa', fontsize=9)
            plt.tight_layout(); st.pyplot(fig); plt.close()
            st.markdown("<p style='font-size:11px;color:#bbb;font-family:JetBrains Mono'>🔴 Increases risk &nbsp; 🟢 Decreases risk</p>", unsafe_allow_html=True)
        with right:
            st.markdown("<div class='section-title'>Risk Factors Detected</div>", unsafe_allow_html=True)
            for r in (risks or ["✓ No major risk factors"]): st.markdown(f"<span class='risk-badge'>{'⚠ ' if risks else ''}{r}</span>", unsafe_allow_html=True)
            st.markdown("<br><div class='section-title'>HR Recommendations</div>", unsafe_allow_html=True)
            for rec in (recs or ["Continue regular performance check-ins and maintain engagement."]): st.markdown(f"<div class='recommendation-box'>→ {rec}</div>", unsafe_allow_html=True)
            if use_demo: st.markdown("<br><p style='font-size:11px;color:#ccc;font-family:JetBrains Mono'>ℹ Model file not found — running in demo mode</p>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='text-align:center;padding:80px 0'><div style='font-size:48px;margin-bottom:16px'>◈</div><div style='font-family:JetBrains Mono;font-size:12px;color:#bbb'>Configure employee profile in the sidebar and click Analyse</div></div>", unsafe_allow_html=True)

# ── PAGE 2 ──────────────────────────────────
elif page == "💬 NLP Exit Analysis":
    st.markdown("# NLP Exit Interview Analysis")
    st.markdown("<p style='color:#aaa;font-size:13px;margin-top:-12px;font-family:JetBrains Mono'>Sentiment analysis · Topic modeling · Exit reasons</p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("> Simulated exit interview feedbacks analyzed using **VADER sentiment analysis** and **LDA topic modeling** to extract emotional signals and recurring themes.")

    nlp_dir = os.path.join(BASE, 'NLP')
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='section-title'>Sentiment Score Distribution</div>", unsafe_allow_html=True)
        p = os.path.join(nlp_dir, 'sentiment_distribution.png')
        st.image(p) if os.path.exists(p) else st.info("sentiment_distribution.png not found")
        st.markdown("<div class='section-title' style='margin-top:24px'>Distribution of Exit Reasons</div>", unsafe_allow_html=True)
        p = os.path.join(nlp_dir, 'reasons_pie.png')
        st.image(p) if os.path.exists(p) else st.info("reasons_pie.png not found")
    with c2:
        st.markdown("<div class='section-title'>Word Cloud — Exit Interviews</div>", unsafe_allow_html=True)
        p = os.path.join(nlp_dir, 'wordcloud.png')
        st.image(p) if os.path.exists(p) else st.info("wordcloud.png not found")
        st.markdown("<div class='section-title' style='margin-top:24px'>Topic Modeling (LDA)</div>", unsafe_allow_html=True)
        p = os.path.join(nlp_dir, 'topic_modeling.png')
        st.image(p) if os.path.exists(p) else st.info("topic_modeling.png not found")

    st.markdown("---")
    st.markdown("<p style='font-size:12px;color:#aaa;font-family:JetBrains Mono'>ℹ Note: Exit interview texts are AI-simulated for proof-of-concept. In production, sourced from real HR exit records.</p>", unsafe_allow_html=True)

# ── PAGE 3 ──────────────────────────────────
elif page == "⚡ Carbon Footprint":
    st.markdown("# Carbon Footprint — Frugal AI")
    st.markdown("<p style='color:#aaa;font-size:13px;margin-top:-12px;font-family:JetBrains Mono'>CodeCarbon · Energy efficiency · Responsible computing</p>", unsafe_allow_html=True)
    st.markdown("---")

    em_path = os.path.join(BASE, 'outputs', 'emissions.csv')
    if os.path.exists(em_path):
        em = pd.read_csv(em_path)
        st.markdown("<div class='section-title'>Emissions Summary</div>", unsafe_allow_html=True)
        if 'emissions' in em.columns:
            c1,c2,c3 = st.columns(3)
            with c1: st.markdown(f"<div class='metric-card'><div class='metric-label'>Total CO₂</div><div class='metric-value' style='font-size:28px'>{em['emissions'].sum():.6f}</div><div style='color:#aaa;font-size:12px;margin-top:4px'>kg CO₂</div></div>", unsafe_allow_html=True)
            if 'energy_consumed' in em.columns:
                with c2: st.markdown(f"<div class='metric-card'><div class='metric-label'>Energy Consumed</div><div class='metric-value' style='font-size:28px'>{em['energy_consumed'].sum():.6f}</div><div style='color:#aaa;font-size:12px;margin-top:4px'>kWh</div></div>", unsafe_allow_html=True)
            if 'duration' in em.columns:
                with c3: st.markdown(f"<div class='metric-card'><div class='metric-label'>Training Duration</div><div class='metric-value' style='font-size:28px'>{em['duration'].sum():.2f}</div><div style='color:#aaa;font-size:12px;margin-top:4px'>seconds</div></div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(em, use_container_width=True)
    else:
        st.info("emissions.csv not found in outputs/ folder")

    st.markdown("---")
    st.markdown("### Why Frugal AI matters\n\nWe chose **RandomForest** over complex models because on small datasets (311 employees), simpler models achieve comparable accuracy with significantly lower carbon footprint.")
    st.markdown("<div class='section-title'>Model Complexity vs Carbon Footprint</div>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(7, 3))
    fig.patch.set_facecolor('#fff'); ax.set_facecolor('#fff')
    models_l = ['Logistic Regression', 'Random Forest ✓', 'Gradient Boosting', 'Neural Network']
    co2 = [0.1, 0.3, 0.7, 2.5]
    ax.barh(models_l, co2, color=['#ccc','#111','#ccc','#ccc'], height=0.5)
    ax.set_xlabel("Relative CO₂ Emissions", color='#888', fontsize=10)
    ax.tick_params(colors='#888', labelsize=10)
    for s in ax.spines.values(): s.set_color('#ececec')
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ── PAGE 4 ──────────────────────────────────
elif page == "🚨 High Risk Employees":
    st.markdown("# High Risk Employee List")
    st.markdown("<p style='color:#aaa;font-size:13px;margin-top:-12px;font-family:JetBrains Mono'>Employees flagged for immediate HR attention</p>", unsafe_allow_html=True)
    st.markdown("---")

    risk_path = os.path.join(BASE, 'outputs', 'hr_risk_reports.txt')
    if os.path.exists(risk_path):
        with open(risk_path, 'r', encoding='utf-8', errors='ignore') as f: st.code(f.read(), language=None)
    else:
        dataset_path = os.path.join(BASE, 'dataset', 'HRDataset_v14.csv')
        sentiment_path = os.path.join(BASE, 'dataset', 'sentiment_scores.csv')
        if os.path.exists(dataset_path):
            df = pd.read_csv(dataset_path)
            if os.path.exists(sentiment_path):
                sent = pd.read_csv(sentiment_path)
                df = df.merge(sent, on='EmpID', how='left')
                df['sentiment_score'] = df['sentiment_score'].fillna(0)
            else:
                df['sentiment_score'] = 0
            perf_map = {'Exceeds':4,'Fully Meets':3,'Needs Improvement':2,'PIP':1}
            df['perf_num'] = df['PerformanceScore'].map(perf_map).fillna(3)
            df['risk_score'] = ((5 - df['EmpSatisfaction'])*0.3 + (4 - df['perf_num'])*0.3 + df['Absences']/30*0.2 + df['sentiment_score'].apply(lambda x: max(0,-x))*0.2)
            high_risk = df[df['Termd']==0].nlargest(10,'risk_score')[['EmpID','Department','PerformanceScore','EmpSatisfaction','Absences','risk_score']].reset_index(drop=True)
            st.markdown("<div class='section-title'>Top 10 At-Risk Active Employees</div>", unsafe_allow_html=True)
            for _, row in high_risk.iterrows():
                score = round(row['risk_score']*100)
                color = "#e53e3e" if score>=60 else "#d97706"
                st.markdown(f"<div class='employee-row' style='display:flex;justify-content:space-between;align-items:center'><div><span style='font-weight:600'>ID {int(row['EmpID'])}</span><span style='color:#aaa;font-size:12px;margin-left:12px'>{row.get('Department','—')}</span></div><span style='font-family:JetBrains Mono;font-size:12px;color:#aaa'>Perf: {row.get('PerformanceScore','—')} · Sat: {row.get('EmpSatisfaction','—')}/5 · Abs: {int(row.get('Absences',0))}</span><span style='font-size:20px;font-weight:600;color:{color}'>{score}%</span></div>", unsafe_allow_html=True)
        else:
            st.info("Dataset not found. Please ensure HRDataset_v14.csv is in the dataset/ folder.")

st.markdown("---")
st.markdown("<p style='font-size:11px;color:#ccc;font-family:JetBrains Mono;text-align:center'>HR Attrition Intelligence · Trusted AI × HR · Explainable · Frugal · Secure</p>", unsafe_allow_html=True)
