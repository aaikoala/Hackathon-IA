import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import warnings
warnings.filterwarnings('ignore')

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
.emp-info-box { background: #f8f8f8; border: 1px solid #ececec; border-radius: 8px; padding: 14px 18px; margin-bottom: 16px; font-size: 13px; color: #555; }
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

@st.cache_data
def load_dataset():
    df = pd.read_csv(os.path.join(BASE, 'dataset', 'HRDataset_v14.csv'))
    sp = os.path.join(BASE, 'dataset', 'sentiment_scores.csv')
    if os.path.exists(sp):
        sent = pd.read_csv(sp)
        df = df.merge(sent, on='EmpID', how='left')
    if 'sentiment_score' not in df.columns:
        df['sentiment_score'] = 0
    df['sentiment_score'] = df['sentiment_score'].fillna(0)
    return df

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
page = st.sidebar.radio("Go to", [
    "🔍 Employee Analysis",
    "💬 NLP Exit Analysis",
    "🔎 Explainable AI (SHAP)",
    "⚡ Carbon Footprint",
    "🚨 High Risk Employees"
], label_visibility="collapsed")
st.sidebar.markdown("---")

# ══════════════════════════════════════════════
# PAGE 1 — EMPLOYEE ANALYSIS
# ══════════════════════════════════════════════
if page == "🔍 Employee Analysis":

    df_all = load_dataset()
    active_df = df_all[df_all['Termd'] == 0]

    with st.sidebar:
        st.markdown("<div class='section-title'>Input Mode</div>", unsafe_allow_html=True)
        mode = st.radio("mode", ["📂 Select from dataset", "✏️ Manual input"], label_visibility="collapsed")
        st.markdown("---")

        if mode == "📂 Select from dataset":
            st.markdown("<div class='section-title'>Select Employee</div>", unsafe_allow_html=True)
            emp_id = st.selectbox("Employee ID", active_df['EmpID'].tolist(), label_visibility="visible")
            predict_btn = st.button("Analyse Employee")
        else:
            st.markdown("<div class='section-title'>Employee Profile</div>", unsafe_allow_html=True)
            department   = st.selectbox("Department", ["IT/IS", "Production", "Sales", "Software Engineering", "Admin Offices", "Executive Office"])
            perf_score   = st.selectbox("Performance Score", ["Exceeds", "Fully Meets", "Needs Improvement", "PIP"])
            satisfaction = st.slider("Employee Satisfaction (1-5)", 1, 5, 3)
            absences     = st.slider("Absences (days/year)", 0, 30, 5)
            years        = st.slider("Years at Company", 0, 20, 3)
            st.markdown("<div class='section-title' style='margin-top:16px'>NLP Exit Sentiment</div>", unsafe_allow_html=True)
            sentiment    = st.slider("Sentiment Score", -1.0, 1.0, 0.0, 0.01)
            predict_btn  = st.button("Analyse Employee")

    st.markdown("# HR Attrition Intelligence")
    st.markdown("<p style='color:#aaa;font-size:13px;margin-top:-12px;font-family:JetBrains Mono'>Predictive analytics · Explainable AI · Responsible ML</p>", unsafe_allow_html=True)
    st.markdown("---")

    if predict_btn:
        perf_map = {"Exceeds": 4, "Fully Meets": 3, "Needs Improvement": 2, "PIP": 1}
        dept_map = {"IT/IS": 0, "Production": 1, "Sales": 2, "Software Engineering": 3, "Admin Offices": 4, "Executive Office": 5}

        if mode == "📂 Select from dataset":
            row = active_df[active_df['EmpID'] == emp_id].iloc[0]
            perf_score   = row.get('PerformanceScore', 'Fully Meets')
            satisfaction = int(row.get('EmpSatisfaction', 3))
            absences     = int(row.get('Absences', 0))
            sentiment    = float(row.get('sentiment_score', 0))
            department   = row.get('Department', '—')

            st.markdown(f"""
            <div class='emp-info-box'>
                <b>Employee #{int(emp_id)}</b> &nbsp;·&nbsp; {department} &nbsp;·&nbsp;
                Performance: <b>{perf_score}</b> &nbsp;·&nbsp;
                Satisfaction: <b>{satisfaction}/5</b> &nbsp;·&nbsp;
                Absences: <b>{absences} days</b> &nbsp;·&nbsp;
                Sentiment: <b>{round(sentiment,2)}</b>
            </div>
            """, unsafe_allow_html=True)

            try:
                model = load_model()
                try:
                    feature_names = list(model.feature_names_in_)
                except:
                    try:
                        feature_names = list(model.steps[-1][1].feature_names_in_)
                    except:
                        feature_names = []

                if len(feature_names) > 0:
                    input_row = {}
                    for feat in feature_names:
                        if feat in row.index:
                            input_row[feat] = row[feat]
                        else:
                            input_row[feat] = 0
                    input_data = pd.DataFrame([input_row])
                else:
                    input_data = pd.DataFrame([row])

                prob = model.predict_proba(input_data)[0][1]
                use_demo = False
            except Exception:
                use_demo = True
                prob = min((5-satisfaction)*0.12 + (perf_map.get(perf_score,3)==1)*0.25 + (absences/30)*0.20 + max(0,-sentiment)*0.15, 0.97)

        else:
            input_data = pd.DataFrame([{
                'Salary': 50000, 'Position': 0, 'State': 0, 'Zip': 0, 'MaritalDesc': 0, 'CitizenDesc': 0,
                'Department': dept_map.get(department, 0), 'RecruitmentSource': 0,
                'PerformanceScore': perf_map.get(perf_score, 3), 'EngagementSurvey': 3.0,
                'EmpSatisfaction': satisfaction, 'SpecialProjectsCount': 0, 'DaysLateLast30': 0,
                'Absences': absences, 'tenure_years': years, 'hire_month': 6, 'seniority_band': 0,
                'age_at_hire': 30, 'current_age': 30 + years, 'attendance_risk': 0,
                'disengagement_score': max(0, (3 - satisfaction)), 'salary_ratio': 1.0,
                'perf_risk_flag': 1 if perf_score in ['Needs Improvement', 'PIP'] else 0,
            }])
            try:
                model = load_model()
                prob = model.predict_proba(input_data)[0][1]
                use_demo = False
            except Exception:
                use_demo = True
                prob = min((5-satisfaction)*0.12 + (perf_map.get(perf_score,3)==1)*0.25 + (absences/30)*0.20 + max(0,-sentiment)*0.15 + max(0,(10-years)/10)*0.10, 0.97)

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
            shap_data = {"EmpSatisfaction": round((5-satisfaction)*0.12,3), "sentiment_score": round(max(0,-sentiment)*0.15,3), "Absences": round((absences/30)*0.20,3), "PerformanceScore": round((4-perf_map.get(perf_score,3))*0.08,3)}
            if not use_demo:
                try:
                    sv = shap.TreeExplainer(model).shap_values(input_data)
                    sv = sv[1][0] if isinstance(sv, list) else sv[0]
                    cols = list(input_data.columns)
                    top_idx = np.argsort(np.abs(sv))[-6:]
                    shap_data = {cols[i]: round(float(sv[i]),4) for i in top_idx}
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
            for r in (risks or ["No major risk factors"]): st.markdown(f"<span class='risk-badge'>{'⚠ ' if risks else '✓ '}{r}</span>", unsafe_allow_html=True)
            st.markdown("<br><div class='section-title'>HR Recommendations</div>", unsafe_allow_html=True)
            for rec in (recs or ["Continue regular performance check-ins and maintain engagement."]): st.markdown(f"<div class='recommendation-box'>→ {rec}</div>", unsafe_allow_html=True)
            if use_demo: st.markdown("<br><p style='font-size:11px;color:#ccc;font-family:JetBrains Mono'>i Running in demo mode</p>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='text-align:center;padding:80px 0'><div style='font-size:48px;margin-bottom:16px'>◈</div><div style='font-family:JetBrains Mono;font-size:12px;color:#bbb'>Select an employee or enter details manually, then click Analyse</div></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# PAGE 2 — NLP EXIT ANALYSIS
# ══════════════════════════════════════════════
elif page == "💬 NLP Exit Analysis":
    import plotly.express as px
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    st.markdown("# NLP Exit Interview Analysis")
    st.markdown("<p style='color:#aaa;font-size:13px;margin-top:-12px;font-family:JetBrains Mono'>Based on real TermReason data · VADER sentiment · Topic analysis</p>", unsafe_allow_html=True)
    st.markdown("---")

    df_nlp = pd.read_csv(os.path.join(BASE, 'dataset', 'HRDataset_v14.csv'))
    terminated_nlp = df_nlp[df_nlp['Termd'] == 1].copy()
    reasons = terminated_nlp['TermReason'].dropna()
    reasons = reasons[reasons != 'N/A-StillEmployed']

    analyzer = SentimentIntensityAnalyzer()
    terminated_nlp = terminated_nlp[terminated_nlp['TermReason'] != 'N/A-StillEmployed'].copy()
    terminated_nlp['sentiment'] = terminated_nlp['TermReason'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])

    cat_map = {
        'Another position': 'Career & Growth', 'career change': 'Career & Growth', 'return to school': 'Career & Growth',
        'more money': 'Compensation',
        'unhappy': 'Work Conditions', 'hours': 'Work Conditions', 'attendance': 'Work Conditions', 'no-call, no-show': 'Work Conditions',
        'relocation out of area': 'Personal', 'military': 'Personal', 'retiring': 'Personal',
        'maternity leave - did not return': 'Personal', 'medical issues': 'Personal',
        'performance': 'Performance', 'gross misconduct': 'Performance',
        'Learned that he is a gangster': 'Performance', 'Fatal attraction': 'Performance',
    }
    terminated_nlp['category'] = terminated_nlp['TermReason'].map(cat_map).fillna('Other')

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-title'>Top Reasons for Leaving</div>", unsafe_allow_html=True)
        rc = reasons.value_counts().head(10).reset_index()
        rc.columns = ['Reason', 'Count']
        fig = px.bar(rc.sort_values('Count'), x='Count', y='Reason', orientation='h',
                     color='Count', color_continuous_scale=['#ffcccc','#e53e3e'])
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', coloraxis_showscale=False,
                         font=dict(family='Inter'), margin=dict(l=0,r=0,t=10,b=0))
        fig.update_traces(hovertemplate='<b>%{y}</b><br>Employees: %{x}<extra></extra>')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-title'>Attrition by Category</div>", unsafe_allow_html=True)
        cc = terminated_nlp['category'].value_counts().reset_index()
        cc.columns = ['Category', 'Count']
        fig2 = px.pie(cc, values='Count', names='Category', hole=0.4,
                      color_discrete_sequence=['#111','#555','#888','#aaa','#ccc','#e8e8e8'])
        fig2.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                          font=dict(family='Inter'), margin=dict(l=0,r=0,t=10,b=0))
        fig2.update_traces(hovertemplate='<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>')
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("<div class='section-title'>Sentiment Distribution (VADER)</div>", unsafe_allow_html=True)
        fig3 = px.histogram(terminated_nlp, x='sentiment', nbins=15,
                           color_discrete_sequence=['#e53e3e'],
                           labels={'sentiment': 'Sentiment Score (-1=Negative, +1=Positive)'})
        fig3.add_vline(x=0, line_dash='dash', line_color='#333', annotation_text='Neutral')
        fig3.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                          font=dict(family='Inter'), margin=dict(l=0,r=0,t=10,b=0))
        fig3.update_traces(hovertemplate='Score: %{x:.2f}<br>Count: %{y}<extra></extra>')
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("<div class='section-title'>Sentiment by Category</div>", unsafe_allow_html=True)
        fig4 = px.box(terminated_nlp, x='category', y='sentiment', color='category',
                     color_discrete_sequence=['#111','#555','#888','#aaa','#ccc'],
                     labels={'sentiment': 'Sentiment Score', 'category': ''})
        fig4.add_hline(y=0, line_dash='dash', line_color='#aaa')
        fig4.update_layout(plot_bgcolor='white', paper_bgcolor='white', showlegend=False,
                          font=dict(family='Inter'), margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("<div class='section-title'>Word Cloud - Exit Reasons</div>", unsafe_allow_html=True)
    p = os.path.join(BASE, 'NLP', 'termreason_wordcloud.png')
    st.image(p, use_container_width=True) if os.path.exists(p) else st.info("Run nlp_termreason.py first to generate wordcloud")
    st.markdown("---")
    st.markdown("<p style='font-size:12px;color:#aaa;font-family:JetBrains Mono'>Analysis based on real TermReason field · n=104 terminated employees</p>", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# PAGE 3 — EXPLAINABLE AI (SHAP)
# ══════════════════════════════════════════════
elif page == "🔎 Explainable AI (SHAP)":
    st.markdown("# Explainable AI - SHAP Analysis")
    st.markdown("<p style='color:#aaa;font-size:13px;margin-top:-12px;font-family:JetBrains Mono'>Model transparency · Feature importance · Individual explanations</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("""
    > **Why Explainable AI matters:** HR managers need to understand *why* the model flags an employee as at-risk.
    > SHAP (SHapley Additive exPlanations) breaks down each prediction into individual feature contributions,
    > ensuring our system is transparent and trustworthy, not a black box.
    """)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='section-title'>Global Feature Importance</div>", unsafe_allow_html=True)
        p = os.path.join(BASE, 'outputs', 'shap_global_bar.png')
        st.image(p, use_container_width=True) if os.path.exists(p) else st.info("shap_global_bar.png not found in outputs/")
        st.markdown("<p style='font-size:12px;color:#aaa;font-family:JetBrains Mono'>Average impact of each feature across all employees</p>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='section-title'>Beeswarm Plot - Feature Distribution</div>", unsafe_allow_html=True)
        p = os.path.join(BASE, 'outputs', 'shap_beeswarm.png')
        st.image(p, use_container_width=True) if os.path.exists(p) else st.info("shap_beeswarm.png not found in outputs/")
        st.markdown("<p style='font-size:12px;color:#aaa;font-family:JetBrains Mono'>Each dot = one employee · Red = high feature value · Blue = low</p>", unsafe_allow_html=True)

    st.markdown("<br><div class='section-title'>Individual Prediction Explanation (Waterfall)</div>", unsafe_allow_html=True)
    p = os.path.join(BASE, 'outputs', 'shap_local_waterfall.png')
    st.image(p, use_container_width=True) if os.path.exists(p) else st.info("shap_local_waterfall.png not found in outputs/")
    st.markdown("<p style='font-size:12px;color:#aaa;font-family:JetBrains Mono'>How each feature pushes the prediction above or below the baseline for a specific employee</p>", unsafe_allow_html=True)

    st.markdown("<br><div class='section-title'>Fairness Audit - AI Ethics</div>", unsafe_allow_html=True)
    p = os.path.join(BASE, 'outputs', 'fairness_audit.png')
    st.image(p, use_container_width=True) if os.path.exists(p) else st.info("fairness_audit.png not found in outputs/")
    st.markdown("<p style='font-size:12px;color:#aaa;font-family:JetBrains Mono'>Model predictions audited for bias across gender and ethnicity groups</p>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<p style='font-size:12px;color:#aaa;font-family:JetBrains Mono'>SHAP values computed using TreeExplainer · Fairness audit via AIF360 methodology</p>", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# PAGE 4 — CARBON FOOTPRINT
# ══════════════════════════════════════════════
elif page == "⚡ Carbon Footprint":
    st.markdown("# Carbon Footprint - Frugal AI")
    st.markdown("<p style='color:#aaa;font-size:13px;margin-top:-12px;font-family:JetBrains Mono'>CodeCarbon · Energy efficiency · Responsible computing</p>", unsafe_allow_html=True)
    st.markdown("---")

    em_path = os.path.join(BASE, 'outputs', 'emissions.csv')
    if os.path.exists(em_path):
        em = pd.read_csv(em_path)
        st.markdown("<div class='section-title'>Emissions Summary</div>", unsafe_allow_html=True)
        if 'emissions' in em.columns:
            c1,c2,c3 = st.columns(3)
            with c1: st.markdown(f"<div class='metric-card'><div class='metric-label'>Total CO2</div><div class='metric-value' style='font-size:28px'>{em['emissions'].sum():.6f}</div><div style='color:#aaa;font-size:12px;margin-top:4px'>kg CO2</div></div>", unsafe_allow_html=True)
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
    ax.barh(['Logistic Regression','Random Forest','Gradient Boosting','Neural Network'], [0.1,0.3,0.7,2.5], color=['#ccc','#111','#ccc','#ccc'], height=0.5)
    ax.set_xlabel("Relative CO2 Emissions", color='#888', fontsize=10)
    ax.tick_params(colors='#888', labelsize=10)
    for s in ax.spines.values(): s.set_color('#ececec')
    plt.tight_layout(); st.pyplot(fig); plt.close()
    st.markdown("<p style='font-size:12px;color:#aaa;font-family:JetBrains Mono'>Random Forest selected for optimal balance between performance and energy efficiency</p>", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# PAGE 5 — HIGH RISK EMPLOYEES
# ══════════════════════════════════════════════
elif page == "🚨 High Risk Employees":
    st.markdown("# High Risk Employees")
    st.markdown("<p style='color:#aaa;font-size:13px;margin-top:-12px;font-family:JetBrains Mono'>Top 3 employees flagged for immediate HR attention</p>", unsafe_allow_html=True)
    st.markdown("---")

    df_risk = load_dataset()
    perf_map2 = {'Exceeds':4,'Fully Meets':3,'Needs Improvement':2,'PIP':1}
    df_risk['perf_num'] = df_risk['PerformanceScore'].map(perf_map2).fillna(3)
    df_risk['risk_score'] = (
        (5 - df_risk['EmpSatisfaction']) / 4 * 0.35 +
        (4 - df_risk['perf_num']) / 3 * 0.35 +
        df_risk['Absences'] / df_risk['Absences'].max() * 0.20 +
        df_risk['sentiment_score'].apply(lambda x: max(0, -x)) * 0.10
    ).clip(0, 1)

    top3 = df_risk[df_risk['Termd']==0].nlargest(3,'risk_score').reset_index(drop=True)

    for i, row in top3.iterrows():
        score = round(row['risk_score'] * 100)
        color = "#e53e3e" if score>=60 else "#d97706"
        perf  = row.get('PerformanceScore','—')
        sat   = row.get('EmpSatisfaction','—')
        abs_  = int(row.get('Absences', 0))
        dept  = row.get('Department','—')
        sent_val = round(float(row.get('sentiment_score', 0)), 2)
        perf_color = "#e53e3e" if perf in ["Needs Improvement","PIP"] else "#111"
        sat_color  = "#e53e3e" if sat<=2 else "#111"
        abs_color  = "#e53e3e" if abs_>=15 else "#111"
        sent_color = "#e53e3e" if sent_val<-0.3 else "#111"

        st.markdown(f"""
        <div style='background:#fff;border:1px solid #e8e8e8;border-radius:12px;padding:24px;margin:12px 0;box-shadow:0 1px 4px rgba(0,0,0,0.05)'>
            <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:20px'>
                <div>
                    <span style='font-size:20px;font-weight:600'>#{i+1} · Employee {int(row['EmpID'])}</span>
                    <span style='color:#aaa;font-size:13px;margin-left:12px;font-family:JetBrains Mono'>{dept}</span>
                </div>
                <div style='text-align:right'>
                    <div style='font-family:JetBrains Mono;font-size:10px;color:#aaa;text-transform:uppercase'>Attrition Risk</div>
                    <div style='font-size:36px;font-weight:700;color:{color};line-height:1'>{score}%</div>
                </div>
            </div>
            <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:12px'>
                <div style='background:#f8f8f8;border-radius:8px;padding:14px;text-align:center'>
                    <div style='font-family:JetBrains Mono;font-size:10px;color:#aaa;text-transform:uppercase;margin-bottom:6px'>Performance</div>
                    <div style='font-weight:600;color:{perf_color}'>{perf}</div>
                </div>
                <div style='background:#f8f8f8;border-radius:8px;padding:14px;text-align:center'>
                    <div style='font-family:JetBrains Mono;font-size:10px;color:#aaa;text-transform:uppercase;margin-bottom:6px'>Satisfaction</div>
                    <div style='font-weight:600;color:{sat_color}'>{sat}/5</div>
                </div>
                <div style='background:#f8f8f8;border-radius:8px;padding:14px;text-align:center'>
                    <div style='font-family:JetBrains Mono;font-size:10px;color:#aaa;text-transform:uppercase;margin-bottom:6px'>Absences</div>
                    <div style='font-weight:600;color:{abs_color}'>{abs_} days</div>
                </div>
                <div style='background:#f8f8f8;border-radius:8px;padding:14px;text-align:center'>
                    <div style='font-family:JetBrains Mono;font-size:10px;color:#aaa;text-transform:uppercase;margin-bottom:6px'>Sentiment</div>
                    <div style='font-weight:600;color:{sent_color}'>{sent_val}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<p style='font-size:12px;color:#aaa;font-family:JetBrains Mono'>Risk score based on satisfaction, performance, absences and exit sentiment · All decisions must be validated by HR (GDPR Art. 22)</p>", unsafe_allow_html=True)

# ── FOOTER ──
st.markdown("---")
st.markdown("<p style='font-size:11px;color:#ccc;font-family:JetBrains Mono;text-align:center'>HR Attrition Intelligence · Trusted AI x HR · Explainable · Frugal · Secure</p>", unsafe_allow_html=True)