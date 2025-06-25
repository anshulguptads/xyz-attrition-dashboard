import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import shap

# --- 1. Page Configuration and Branding ---
st.set_page_config(
    page_title="XYZ Workforce Attrition Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🚀 XYZ Employee Attrition Dashboard")
st.markdown("> Proactive churn diagnostics • Predictive retention heuristics • Data-driven interventions")

# --- 2. Data Ingestion & Preprocessing ---
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    # Basic cleaning
    df = df.dropna()
    # Encode categorical features
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])
    return df

data = load_data("EA.csv")

# KPI Metrics
col1, col2, col3 = st.columns(3)
attrition_rate = data['Attrition'].mean()
col1.metric("Attrition Rate", f"{attrition_rate:.1%}")
col2.metric("Total Employees", data.shape[0])
col3.metric("Features Monitored", data.shape[1] - 1)

st.markdown("---")

# --- 3. Attrition Profile Analytics ---
st.subheader("📊 Attrition Profile")
dept_chart = px.histogram(
    data, x="Department", color="Attrition",
    barmode="group", title="Attrition by Department"
)
st.plotly_chart(dept_chart, use_container_width=True)

satisfaction_chart = px.box(
    data, x="Attrition", y="JobSatisfaction",
    points="all", title="Job Satisfaction vs Attrition"
)
st.plotly_chart(satisfaction_chart, use_container_width=True)

# Correlation Heatmap
st.subheader("🔎 Correlation Matrix")
corr = data.corr()
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
ax.set_xticklabels(corr.columns, rotation=45, ha='right')
ax.set_yticklabels(corr.columns)
plt.colorbar(im)
st.pyplot(fig)

st.markdown("---")

# --- 4. Predictive Modeling & Feature Importance ---
st.subheader("🤖 Predictive Attrition Model")

# Split & train
X = data.drop("Attrition", axis=1)
y = data["Attrition"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
score = rf.score(X_test, y_test)
st.write(f"**Model Accuracy:** {score:.2%}")

# SHAP Explanation
st.markdown("#### Model-Driven Attrition Levers (SHAP Values)")
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)
fig_shap = shap.summary_plot(shap_values[1], X_test, plot_type="bar", show=False)
st.pyplot(bbox_inches='tight')

st.markdown("---")

# --- 5. Actionable Recommendations ---
st.subheader("💡 Strategic Retention Recommendations")
st.markdown("""
1. **Elevate Job Satisfaction**  
   – Proactively deploy pulse surveys in departments with elevated churn risk.  
   – Tailor career-path frameworks to high-impact roles (e.g., R&D, Sales) identified above.

2. **Optimize Compensation Parity**  
   – Benchmark pay equity across tenure cohorts; remediate any statistical outliers.  
   – Introduce spot bonuses for high-risk individuals flagged by predictive model.

3. **Enhance Manager-Employee Engagement**  
   – Standardize 1:1 check-ins for mid-level managers with >5 direct reports.  
   – Embed AI-driven coaching nudges via the internal talent platform.

4. **Talent Mobility & Upskilling**  
   – Fast-track rotation programs for demographics with lower tenure satisfaction.  
   – Integrate microlearning modules to shore up critical skill gaps.
""")

# --- 6. Footer & Deployment Note ---
st.markdown("---")
st.caption("© 2025 XYZ Company | Dashboard deployed via Streamlit Cloud | Data refreshed: June 25, 2025")

