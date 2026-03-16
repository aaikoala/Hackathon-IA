import os
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud

# save figures in the same folder as this script
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────
# 1. SIMULATED EXIT INTERVIEW FEEDBACKS
# ─────────────────────────────────────────────
feedbacks = [
    {"id": 1,  "category": "Salary & Management",      "feedback": "I felt my salary was too low compared to market rates. There was no clear promotion path."},
    {"id": 2,  "category": "Salary & Management",      "feedback": "Despite working here for years, my salary never kept up with inflation or my growing responsibilities."},
    {"id": 3,  "category": "Salary & Management",      "feedback": "I discovered colleagues with less experience were earning significantly more than me."},
    {"id": 4,  "category": "Workload & Life Balance",  "feedback": "The workload was unmanageable and I had no work-life balance. I was exhausted every day."},
    {"id": 5,  "category": "Workload & Life Balance",  "feedback": "Constant overtime and weekend work made it impossible to maintain a healthy personal life."},
    {"id": 6,  "category": "Workload & Life Balance",  "feedback": "The stress levels were unbearable. I was burning out and had no support from management."},
    {"id": 7,  "category": "Team & Culture",           "feedback": "Management was unsupportive and communication was very poor within the team."},
    {"id": 8,  "category": "Team & Culture",           "feedback": "My manager micromanaged every task and never trusted me to work independently."},
    {"id": 9,  "category": "Team & Culture",           "feedback": "Leadership made decisions without consulting the team, leaving us confused and demotivated."},
    {"id": 10, "category": "Career Development",       "feedback": "There was no opportunity to grow or learn new skills in my position."},
    {"id": 11, "category": "Career Development",       "feedback": "I asked for a promotion three times and was always told to wait. I stopped believing it would happen."},
    {"id": 12, "category": "Career Development",       "feedback": "The company had no training programs and I felt my skills were stagnating."},
    {"id": 13, "category": "Team & Culture",           "feedback": "The company culture did not match my values. I felt isolated and unmotivated."},
    {"id": 14, "category": "Team & Culture",           "feedback": "There was a toxic atmosphere in the office. Gossip and politics were exhausting."},
    {"id": 15, "category": "Team & Culture",           "feedback": "I never felt like I belonged here. The team was not welcoming to new ideas."},
    {"id": 16, "category": "Salary & Management",      "feedback": "I felt undervalued despite consistently good performance reviews over the years."},
    {"id": 17, "category": "Salary & Management",      "feedback": "My contributions were never acknowledged. It felt like my work was invisible."},
    {"id": 18, "category": "Salary & Management",      "feedback": "I worked hard every day but received no recognition or appreciation from anyone."},
    {"id": 19, "category": "Career Development",       "feedback": "I received a better offer elsewhere with higher salary and more responsibilities."},
    {"id": 20, "category": "Career Development",       "feedback": "A competitor offered me a role that aligned much better with my career goals."},
    {"id": 21, "category": "Team & Culture",           "feedback": "Conflicts with my direct manager made it impossible to continue working here."},
    {"id": 22, "category": "Team & Culture",           "feedback": "The relationship with my team deteriorated over time and became very uncomfortable."},
    {"id": 23, "category": "Team & Culture",           "feedback": "I experienced discrimination that was never addressed by HR despite my complaints."},
    {"id": 24, "category": "Career Development",       "feedback": "My role became repetitive and unchallenging. I needed more stimulating work."},
    {"id": 25, "category": "Career Development",       "feedback": "The projects I was assigned to had no impact and I felt my time was wasted."},
    {"id": 26, "category": "Workload & Life Balance",  "feedback": "I decided to pursue further education to advance my career in a different direction."},
    {"id": 27, "category": "Workload & Life Balance",  "feedback": "I relocated to another city for family reasons and remote work was not an option."},
    {"id": 28, "category": "Workload & Life Balance",  "feedback": "I enjoyed the team but the commute and long hours were unsustainable long term."},
    {"id": 29, "category": "Career Development",       "feedback": "It was a difficult decision. I liked my colleagues but the opportunity was too good to pass up."},
    {"id": 30, "category": "Career Development",       "feedback": "I have mostly positive memories here but felt it was time for a new challenge."},
]

feedback_df = pd.DataFrame(feedbacks)
texts = feedback_df['feedback'].tolist()

# ─────────────────────────────────────────────
# 2. VADER SENTIMENT ANALYSIS
# ─────────────────────────────────────────────
analyzer = SentimentIntensityAnalyzer()
feedback_df['sentiment_score'] = feedback_df['feedback'].apply(
    lambda x: analyzer.polarity_scores(x)['compound']
)
print(feedback_df[['id', 'feedback', 'sentiment_score']])

# ─────────────────────────────────────────────
# 3. MERGE WITH HR DATASET
# ─────────────────────────────────────────────
df = pd.read_csv(r'D:\UserData\Desktop\Hackathon-IA\dataset\HRDataset_v14.csv')

terminated = df[df['Termd'] == 1].head(30).copy()
terminated['sentiment_score'] = feedback_df['sentiment_score'].values

df = df.merge(
    terminated[['EmpID', 'sentiment_score']],
    on='EmpID',
    how='left'
)
df['sentiment_score'] = df['sentiment_score'].fillna(0)

print("\ndataset after merge:")
print(df[['EmpID', 'Termd', 'sentiment_score']].head(20))

# ─────────────────────────────────────────────
# 4. PLOT 1 — SENTIMENT DISTRIBUTION
# ─────────────────────────────────────────────
terminated_only = df[(df['Termd'] == 1) & (df['sentiment_score'] != 0)]

plt.figure()
terminated_only['sentiment_score'].hist(bins=10, color='red')
plt.title('Sentiment Score Distribution - Terminated Employees')
plt.xlabel('Sentiment Score (-1=Negative, +1=Positive)')
plt.ylabel('Count')
plt.axvline(x=0, color='black', linestyle='--', label='Neutral')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'sentiment_distribution.png'))
plt.show()

# ─────────────────────────────────────────────
# 5. PLOT 2 — TOPIC MODELING (LDA)
# ─────────────────────────────────────────────
vectorizer = CountVectorizer(stop_words='english', max_features=50)
X = vectorizer.fit_transform(texts)

lda = LatentDirichletAllocation(n_components=4, random_state=42)
lda.fit(X)

feature_names = vectorizer.get_feature_names_out()
topic_labels = ['Team & Culture', 'Workload & Life Balance', 'Career Development', 'Salary & Management']

print("\n=== Topic Modeling Results ===")
for i, topic in enumerate(lda.components_):
    top_words = [feature_names[j] for j in topic.argsort()[-5:]]
    print(f"Topic {i+1} ({topic_labels[i]}): {', '.join(top_words)}")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for i, (ax, topic) in enumerate(zip(axes.flatten(), lda.components_)):
    top_indices = topic.argsort()[-8:]
    top_words = [feature_names[j] for j in top_indices]
    top_scores = [topic[j] for j in top_indices]
    ax.barh(top_words, top_scores, color='steelblue')
    ax.set_title(f'Topic {i+1}: {topic_labels[i]}')
    ax.set_xlabel('Score')
plt.suptitle('Exit Interview - Topic Modeling (LDA)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'topic_modeling.png'))
plt.show()

# ─────────────────────────────────────────────
# 6. PLOT 3 — WORD CLOUD
# ─────────────────────────────────────────────
all_text = ' '.join(texts)
wc = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(all_text)

plt.figure(figsize=(10, 5))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title('Most Common Words in Exit Interviews', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'wordcloud.png'))
plt.show()

# ─────────────────────────────────────────────
# 7. PLOT 4 — REASONS PIE CHART
# ─────────────────────────────────────────────
category_counts = feedback_df['category'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(
    category_counts.values,
    labels=category_counts.index,
    autopct='%1.0f%%',
    colors=['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
)
plt.title('Distribution of Reasons for Leaving', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'reasons_pie.png'))
plt.show()

# ─────────────────────────────────────────────
# 8. HR RECOMMENDATION GENERATOR
# ─────────────────────────────────────────────
def generate_hr_recommendation(emp_id, perf_score, emp_satisfaction, sentiment_score, absences):
    risks = []
    recommendations = []

    # 绩效差 (PerformanceScore: Exceeds/Fully Meets/Needs Improvement/PIP)
    if perf_score in ['Needs Improvement', 'PIP']:
        risks.append("low performance score")
        recommendations.append("schedule a performance improvement plan with clear goals")

    # 满意度低 (EmpSatisfaction: 1-5)
    if emp_satisfaction <= 2:
        risks.append("low employee satisfaction")
        recommendations.append("conduct a 1-on-1 meeting to understand concerns and improve engagement")

    # 情感分数负面
    if sentiment_score < -0.3:
        risks.append("highly negative exit sentiment")
        recommendations.append("review exit interview feedback and address systemic issues")

    # 缺勤多 (Absences)
    if absences >= 15:
        risks.append("high number of absences")
        recommendations.append("check employee wellbeing and consider flexible work arrangements")

    if not risks:
        return f"✅  Employee {emp_id}: Low risk. Continue regular check-ins."

    risk_str = ", ".join(risks)
    rec_str = "; ".join(recommendations)

    return (
        f"⚠️  Employee {emp_id} | Risk factors: {risk_str}.\n"
        f"   → HR Recommendation: {rec_str}."
    )

print("\n=== HR RECOMMENDATIONS ===")
sample = terminated.head(10).merge(
    df[['EmpID', 'sentiment_score']], on='EmpID', how='left'
)

for _, row in sample.iterrows():
    report = generate_hr_recommendation(
        emp_id=row['EmpID'],
        perf_score=row.get('PerformanceScore', 'Fully Meets'),
        emp_satisfaction=row.get('EmpSatisfaction', 3),
        sentiment_score=row.get('sentiment_score_y', row.get('sentiment_score_x', 0)),
        absences=row.get('Absences', 0)
    )
    print(report)
    print()
# 在nlp_module.py最后加这行
terminated[['EmpID', 'sentiment_score']].to_csv(
    r'D:\UserData\Desktop\Hackathon-IA\dataset\sentiment_scores.csv', 
    index=False
)
print("Sentiment scores exported!")
print(f"\nAll figures saved to: {OUTPUT_DIR}")