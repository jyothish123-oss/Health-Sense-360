import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import plotly.express as px
import plotly.graph_objects as go
import os

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Health Sense 360",
    page_icon="🏥",
    layout="wide"
)
# ======================================================
# 🏥 HEALTH SENSE 360  - UI STYLING MODULE
# Purpose: Converts Streamlit default UI into a
#          professional medical SaaS dashboard look
# ======================================================

st.markdown("""
<style>

/* ================= GLOBAL APP THEME ================= */
/* Sets dark medical SaaS background and padding */
body {
    background-color: #0e1117;
}

/* Main app container spacing */
.block-container {
    padding: 2rem;
    max-width: 1400px;
}

/* ================= TYPOGRAPHY ================= */
/* Main title styling */
h1 {
    color: #4fd1c5;  /* medical teal */
    font-weight: 800;
    letter-spacing: 1px;
}

/* Section headings */
h2, h3 {
    color: #e2e8f0;
}

/* ================= SIDEBAR DESIGN ================= */
/* Sidebar background gradient for clinical panel */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a, #111827);
    border-right: 1px solid #1f2937;
}

/* Sidebar text color consistency */
[data-testid="stSidebar"] * {
    color: #e5e7eb;
}

/* Sidebar card UI for grouping elements */
.sidebar-card {
    background: rgba(255,255,255,0.05);
    padding: 12px;
    border-radius: 12px;
    margin-bottom: 10px;
    border: 1px solid rgba(255,255,255,0.08);
}

/* ================= METRICS CARDS ================= */
/* Enhances Streamlit metric widgets */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    padding: 15px;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.08);
}

/* Metric values highlight */
[data-testid="stMetricValue"] {
    color: #4fd1c5;
}

/* ================= BUTTON STYLING ================= */
/* Primary action buttons (Prediction, etc.) */
.stButton > button {
    background: linear-gradient(90deg, #06b6d4, #3b82f6);
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    border: none;
    font-weight: 600;
    transition: 0.3s;
}

/* Button hover animation */
.stButton > button:hover {
    transform: scale(1.02);
    box-shadow: 0px 4px 20px rgba(59,130,246,0.4);
}

/* ================= CARD COMPONENT ================= */
/* Custom reusable card style */
.card {
    background: rgba(255,255,255,0.04);
    padding: 20px;
    border-radius: 15px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 10px 25px rgba(0,0,0,0.3);
}

/* ================= ALERT BOXES ================= */
/* Success message styling */
.stSuccess {
    background-color: rgba(34,197,94,0.15);
    border-radius: 10px;
    padding: 10px;
}

/* Warning message styling */
.stWarning {
    background-color: rgba(245,158,11,0.15);
    border-radius: 10px;
    padding: 10px;
}

/* Error message styling */
.stError {
    background-color: rgba(239,68,68,0.15);
    border-radius: 10px;
    padding: 10px;
}

/* ================= DATA TABLE UI ================= */
/* Dataframe container styling */
[data-testid="stDataFrame"] {
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.08);
}

/* ================= TAB DESIGN ================= */
/* Tab navigation container spacing */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}

/* Individual tab styling */
.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.05);
    border-radius: 10px;
    padding: 10px 16px;
    font-weight: 600;
}

/* Active tab highlight */
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #06b6d4, #3b82f6);
    color: white;
}

/* ================= PLOT AREA ================= */
/* Plotly chart container styling */
.js-plotly-plot {
    border-radius: 12px;
    overflow: hidden;
    background: rgba(255,255,255,0.02);
    padding: 10px;
}

</style>
""", unsafe_allow_html=True)

# ======================================================
# END OF UI STYLING MODULE
# ======================================================

# ---------------- DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("diabetes.csv")

df = load_data()

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

# ---------------- SESSION ----------------
if "history" not in st.session_state:
    st.session_state.history = []   # ✅ FIX: initialize list first


# ======================================================
# 💾 REPORT STORAGE SYSTEM
# ======================================================
import os

REPORT_FILE = "saved_reports.csv"

# Create file if not exists
if not os.path.exists(REPORT_FILE):
    pd.DataFrame(columns=[
        "Date", "Age", "BMI", "Glucose", "BP",
        "Risk", "Wellness", "ChronicRisk"
    ]).to_csv(REPORT_FILE, index=False)

# ======================================================
# 🏥 PROFESSIONAL SIDEBAR
# ======================================================

with st.sidebar:

    st.markdown("## 🏥 Clinical Control Panel")

    # ---------------- PATIENT ID CARD ----------------
    st.markdown("### 👤 Patient Identity")

    patient_name = st.text_input("Patient Name", "John Doe")
    patient_id = st.text_input("Patient ID", "HS360-0001")

    st.success(f"🟢 Active Profile: {patient_name}")

    st.markdown("---")

    # ---------------- MODE SELECTION ----------------
    st.markdown("### ⚙ System Mode")

    mode = st.selectbox(
        "Select Role",
        ["Doctor Mode 🧑‍⚕️", "Analyst Mode 📊", "Patient Mode 🧍"]
    )

    if "Doctor" in mode:
        st.info("Full diagnostic access enabled")
    elif "Analyst" in mode:
        st.warning("Data visualization mode active")
    else:
        st.success("Simplified patient view")

    st.markdown("---")

    # ---------------- CLINICAL INPUTS ----------------
    st.markdown("### 🧬 Clinical Vitals")

    with st.expander("Vital Parameters", expanded=True):
        preg = st.number_input("Pregnancies", 0, 20, 1)
        glucose = st.number_input("Glucose", 0, 300, 120)
        bp = st.number_input("Blood Pressure", 0, 200, 80)
        skin = st.number_input("Skin Thickness", 0, 100, 20)
        insulin = st.number_input("Insulin", 0, 900, 80)

    with st.expander("Body Metrics", expanded=True):
        bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
        dpf = st.number_input("Diabetes Pedigree", 0.0, 3.0, 0.5)
        age = st.number_input("Age", 1, 100, 30)

    with st.expander("Physical Stats", expanded=True):
        height = st.number_input("Height (cm)", 100, 220, 170)
        weight = st.number_input("Weight (kg)", 20, 200, 70)

    st.markdown("---")

    # ---------------- LIVE HEALTH SNAPSHOT ----------------
    st.markdown("### 📊 Live Health Snapshot")

    bmi_calc = round(weight / ((height / 100) ** 2), 2)

    col1, col2 = st.columns(2)
    col1.metric("BMI", bmi_calc)
    col2.metric("Age", age)

    # Risk preview (simple heuristic BEFORE prediction)
    preview_risk = 0

    if glucose > 140:
        preview_risk += 30
    if bmi_calc > 25:
        preview_risk += 25
    if bp > 90:
        preview_risk += 20
    if age > 40:
        preview_risk += 15

    st.markdown("### ⚠ Risk Preview")
    st.progress(min(preview_risk, 100))

    if preview_risk < 30:
        st.success("Low Risk Profile 🟢")
    elif preview_risk < 60:
        st.warning("Moderate Risk 🟠")
    else:
        st.error("High Risk 🔴")

    st.markdown("---")

    # ---------------- SYSTEM STATUS ----------------
    st.markdown("### ⚙ System Status")

    st.success("AI Engine: Online")
    st.info("Model: Random Forest")
    st.caption("All systems operational")

# ---------------- CALCULATIONS ----------------
bmi_calc = round(weight / ((height / 100) ** 2), 2)
water = round(weight * 0.035, 2)
calories = round((10 * weight) + (6.25 * height) - (5 * age) + 5)

# ---------------- HEADER ----------------
st.title("🏥 Health Sense 360")
st.markdown("### Advanced Clinical AI Decision Support System")
st.markdown("---")

# ======================================================
# 📊 TAB 1 - PATIENT OVERVIEW 
# ======================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "🧠 AI Diagnosis",
    "📈 Analytics",
    "📜 Report",
    "🍽 Lifestyle",
    "⚙ System"
])

with tab1:

    st.subheader("🏥 Patient Clinical Intelligence Dashboard")

    # ======================================================
    # 📊 CORE PATIENT METRICS
    # ======================================================
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Age", age, "Years")
    col2.metric("BMI", bmi_calc, "Body Index")
    col3.metric("Water Intake", f"{water} L", "Daily Hydration")
    col4.metric("Calories", f"{calories}", "Energy Need")

    st.markdown("---")

    # ======================================================
    # 🧠 CLINICAL RISK ENGINE
    # ======================================================
    risk_score = 0

    if glucose > 140:
        risk_score += 30
    if bmi_calc > 25:
        risk_score += 25
    if bp > 90:
        risk_score += 20
    if age > 40:
        risk_score += 15

    risk_score = min(risk_score, 100)

    # ======================================================
    # 🏥 PROFESSIONAL RISK CLASSIFICATION 
    # ======================================================
    if risk_score <= 20:
        risk_label = "🟢 Physiologically Stable Profile"
        risk_state = "All biomarkers within normal clinical range"
        ui_type = "success"

    elif risk_score <= 40:
        risk_label = "🟡 Early Risk Indicators Detected"
        risk_state = "Preventive monitoring recommended"
        ui_type = "info"

    elif risk_score <= 70:
        risk_label = "🟠 Moderate Clinical Risk"
        risk_state = "Lifestyle intervention required"
        ui_type = "warning"

    else:
        risk_label = "🔴 High Clinical Risk Condition"
        risk_state = "Immediate clinical evaluation advised"
        ui_type = "error"

    # ======================================================
    # 📊 RISK DASHBOARD PANEL
    # ======================================================
    st.markdown("### 🧠 Clinical Risk Stratification Panel")

    colA, colB = st.columns(2)

    with colA:

        if ui_type == "success":
            st.success(risk_label)
        elif ui_type == "info":
            st.info(risk_label)
        elif ui_type == "warning":
            st.warning(risk_label)
        else:
            st.error(risk_label)

        st.caption(risk_state)

        st.progress(risk_score / 100)

        st.markdown(f"""
        **Clinical Interpretation**
        - Risk Score: `{risk_score}/100`
        - Status Level: `{risk_label}`
        - Recommendation: `{risk_state}`
        """)

    with colB:

        import plotly.graph_objects as go

        fig = go.Figure(data=[
            go.Pie(
                labels=["Clinical Risk", "Health Reserve"],
                values=[risk_score, 100 - risk_score],
                hole=0.5,
                marker_colors=["#ef4444", "#10b981"]
            )
        ])

        fig.update_layout(
            title="Patient Health Balance Index",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e5e7eb"),
            height=350
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ======================================================
    # 📊 ADVANCED VITAL VISUALIZATION 
    # ======================================================
    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 📌 Vital Sign Index")

        vitals_df = pd.DataFrame({
            "Metric": ["BMI", "Glucose", "BP", "Insulin"],
            "Value": [bmi_calc, glucose, bp, insulin]
        })

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=vitals_df["Metric"],
            y=vitals_df["Value"],
            text=vitals_df["Value"],
            textposition="auto",
            marker=dict(
                color=["#06b6d4", "#3b82f6", "#10b981", "#f59e0b"],
                line=dict(color="#ffffff", width=1)
            )
        ))

        fig.update_layout(
            title="Patient Vital Overview",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e5e7eb"),
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        st.markdown("### 📊 Metabolic Health Radar")

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=[bmi_calc, glucose/10, bp, insulin/10],
            theta=["BMI", "Glucose", "BP", "Insulin"],
            fill='toself',
            name="Patient Profile",
            line_color="#06b6d4"
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, max(glucose/10, bp, insulin/10, bmi_calc) + 10])
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e5e7eb"),
            title="Physiological Balance Map"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ======================================================
    # 🧠 CLINICAL INSIGHT SUMMARY PANEL 
    # ======================================================
    st.markdown("### 🧠 Clinical Insight Summary")

    insights = []

    if glucose > 140:
        insights.append(("Metabolic Alert", "Elevated glucose indicates insulin resistance"))
    if bmi_calc > 25:
        insights.append(("Weight Risk", "BMI above optimal clinical range"))
    if bp > 90:
        insights.append(("Cardiovascular Alert", "Blood pressure above recommended level"))
    if age > 40:
        insights.append(("Age Factor", "Age-related metabolic decline risk"))

    if not insights:
        insights.append(("Normal Profile", "All physiological parameters stable"))

    for title, msg in insights:
        st.info(f"🔹 {title}: {msg}")
# ======================================================
# 🧠 TAB 2 - AI DIAGNOSIS 
# ======================================================
with tab2:

    st.subheader("🧠 AI Clinical Diagnostic Engine")

    st.markdown("""
    <div style='padding:12px; border-radius:12px; background:#1f2937'>
    🏥 AI model analyzing metabolic, cardiovascular, and lifestyle indicators in real time
    </div>
    """, unsafe_allow_html=True)

    # ======================================================
    # 🚀 MAIN DIAGNOSTIC TRIGGER
    # ======================================================
    if st.button("🚀 Run Full Diagnostic Scan", use_container_width=True):

        # ---------------- DIABETES PREDICTION ----------------
        input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]
        risk = round(prob * 100, 2)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🧾 Diabetes Diagnosis Result")

            if prediction == 1:
                st.error(f"⚠ High Diabetes Risk: {risk}%")
            else:
                st.success(f"✅ Low Diabetes Risk: {risk}%")

        # ---------------- WELLNESS SCORE ----------------
        wellness = max(
            0,
            100 - (risk * 0.5) - max(0, (bmi_calc - 25) * 2)
        )

        with col2:
            st.markdown("### 🧠 Wellness Score Gauge")

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=wellness,
                title={"text": "Health Index"},
                gauge={"axis": {"range": [0, 100]}}
            ))
            st.plotly_chart(fig, use_container_width=True)

        # ======================================================
        # 🧠 AI INSIGHT ENGINE
        # ======================================================
        st.markdown("### 🧠 AI Insight Dashboard")

        insights = []

        if glucose > 140:
            insights.append(("Glucose Risk", "High sugar levels detected"))
        if bmi_calc > 25:
            insights.append(("BMI Alert", "Overweight condition"))
        if bp > 90:
            insights.append(("BP Warning", "Hypertension risk"))
        if age > 40:
            insights.append(("Age Factor", "Increased metabolic risk"))
        if insulin > 150:
            insights.append(("Insulin Alert", "Possible insulin resistance"))

        if not insights:
            insights.append(("Healthy Profile", "All parameters within normal range"))

        for title, msg in insights:
            st.info(f"🔹 {title}: {msg}")

        st.markdown("---")

        # ======================================================
        # 🧠 CHRONIC DISEASE RISK MODULE
        # ======================================================
        st.subheader("🧠 Chronic Disease Risk Assessment")

        chronic_risk = 0

        if glucose > 140:
            chronic_risk += 25

        if bmi_calc > 30:
            chronic_risk += 25
        elif bmi_calc > 25:
            chronic_risk += 15

        if bp > 90:
            chronic_risk += 20

        if age > 45:
            chronic_risk += 20
        elif age > 35:
            chronic_risk += 10

        if insulin > 150:
            chronic_risk += 15

        chronic_risk = min(chronic_risk, 100)

        col1, col2 = st.columns(2)

        with col1:

            if chronic_risk < 30:
                st.success(f"🟢 Low Chronic Disease Risk: {chronic_risk}%")
                st.caption("Stable metabolic and cardiovascular profile")
            elif chronic_risk < 60:
                st.warning(f"🟠 Moderate Chronic Disease Risk: {chronic_risk}%")
                st.caption("Preventive lifestyle intervention recommended")
            else:
                st.error(f"🔴 High Chronic Disease Risk: {chronic_risk}%")
                st.caption("Immediate clinical attention suggested")

        with col2:

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=chronic_risk,
                title={"text": "Chronic Disease Score"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#ef4444"}
                }
            ))

            st.plotly_chart(fig, use_container_width=True)

        # ======================================================
        # 📊 RISK BREAKDOWN CHART
        # ======================================================
        st.markdown("### 📊 Risk Factor Breakdown")

        breakdown = pd.DataFrame({
            "Factor": ["Glucose", "BMI", "BP", "Age", "Insulin"],
            "Impact": [
                25 if glucose > 140 else 5,
                25 if bmi_calc > 30 else 15,
                20 if bp > 90 else 5,
                20 if age > 45 else 10,
                15 if insulin > 150 else 5
            ]
        })

        fig2 = px.bar(
            breakdown,
            x="Factor",
            y="Impact",
            color="Impact",
            title="Chronic Disease Risk Contributors"
        )

        st.plotly_chart(fig2, use_container_width=True)

        # ======================================================
        # 💾 SAVE TO HISTORY 
        # ======================================================
        st.session_state.history.append({
            "Date": datetime.now(),
            "Age": age,
            "BMI": bmi_calc,
            "Glucose": glucose,
            "BP": bp,
            "Risk": risk,
            "Wellness": round(wellness, 2),
            "ChronicRisk": chronic_risk
        })

# ======================================================
# 📈 TAB 3 - ANALYTICS 
# ======================================================
with tab3:

    st.subheader("📈 Clinical Analytics Center")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Distribution Analysis")

        fig = px.histogram(df, x="Glucose", nbins=25, color_discrete_sequence=["#6366f1"])
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.box(df, y="BMI", color_discrete_sequence=["#10b981"])
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown("### 🔬 Correlation Heatmap")

        fig3 = px.imshow(df.corr(), text_auto=True, color_continuous_scale="Blues")
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    st.markdown("### 📊 Model Performance Status")

    st.success(f"Random Forest Accuracy: {accuracy:.2%}")
# ======================================================
# 📜 TAB 4 - REPORT 
# ======================================================
with tab4:

    st.subheader("📜 Electronic Medical Record Dashboard")

    if st.session_state.history:

        report = pd.DataFrame(st.session_state.history)

        # ================= SUMMARY CARDS =================
        c1, c2, c3 = st.columns(3)

        c1.metric("Total Records", len(report))
        c2.metric("Avg Risk", f"{report['Risk'].mean():.2f}%")
        c3.metric("Avg Wellness", f"{report['Wellness'].mean():.2f}")

        st.markdown("---")

        # ================= VISUAL TREND =================
        col1, col2 = st.columns(2)

        with col1:
            fig = px.line(
                report,
                x="Date",
                y="Wellness",
                markers=True,
                title="Wellness Trend Curve"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.bar(
                report,
                x="Date",
                y="Risk",
                title="Risk Timeline"
            )
            st.plotly_chart(fig2, use_container_width=True)

        # ================= HEAT SUMMARY =================
        st.markdown("### 📊 Patient Health Heatmap View")

        fig3 = px.imshow(
            report[["Risk", "Wellness", "BMI"]],
            text_auto=True,
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig3, use_container_width=True)

        # ================= DOWNLOAD =================
        st.download_button(
            "⬇ Export Medical Report",
            report.to_csv(index=False),
            "medical_report.csv",
            "text/csv"
        )

    else:
        st.info("No clinical records available yet.")

# ======================================================
# 🍽 TAB 5 - LIFESTYLE 
# ======================================================
with tab5:

    st.subheader("🍽 AI Lifestyle Prescription Engine")

    col1, col2 = st.columns(2)

    # ================= DIET CARD =================
    with col1:

        st.markdown("### 🥗 Nutrition Prescription")

        if glucose > 140:

            st.error("Diabetic Care Plan")

            st.markdown("""
            <div class='card'>
            ✔ Low sugar diet recommended<br>
            ✔ Increase fiber intake<br>
            ✔ Avoid processed food
            </div>
            """, unsafe_allow_html=True)

        else:

            st.success("Balanced Nutrition Plan")

            st.markdown("""
            <div class='card'>
            ✔ High protein diet<br>
            ✔ Fruits & vegetables<br>
            ✔ Balanced carbs intake
            </div>
            """, unsafe_allow_html=True)

        # mini pie chart
        fig = px.pie(
            names=["Protein", "Carbs", "Fats"],
            values=[30, 50, 20],
            title="Recommended Macro Split"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ================= ACTIVITY CARD =================
    with col2:

        st.markdown("### 🏃 Activity Prescription")

        if bmi_calc > 30:
            st.error("High Intensity Plan")

            st.markdown("""
            <div class='card'>
            ✔ 45 min brisk walking<br>
            ✔ Strength training<br>
            ✔ Daily cardio
            </div>
            """, unsafe_allow_html=True)

        elif bmi_calc > 25:
            st.warning("Moderate Plan")

            st.markdown("""
            <div class='card'>
            ✔ 30 min jogging<br>
            ✔ Yoga sessions<br>
            ✔ Light cardio
            </div>
            """, unsafe_allow_html=True)

        else:
            st.success("Maintenance Plan")

            st.markdown("""
            <div class='card'>
            ✔ 30 min walking<br>
            ✔ Stretching<br>
            ✔ Active lifestyle
            </div>
            """, unsafe_allow_html=True)

        # activity chart
        fig2 = px.bar(
            x=["Cardio", "Strength", "Flexibility"],
            y=[70, 50, 60],
            title="Fitness Balance Score"
        )
        st.plotly_chart(fig2, use_container_width=True)

# ======================================================
# ⚙ TAB 6 - SYSTEM INFO 
# ======================================================
with tab6:

    st.subheader("⚙ AI System Control Dashboard")

    # ================= STATUS CARDS =================
    col1, col2, col3 = st.columns(3)

    col1.metric("Model", "Random Forest")
    col2.metric("Accuracy", f"{accuracy:.2%}")
    col3.metric("Status", "Active 🟢")

    st.markdown("---")

    # ================= SYSTEM HEALTH CHART =================
    st.markdown("### 🧠 System Performance Monitor")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=[80, 85, 90, 88, 92],
        mode="lines+markers",
        name="CPU Usage"
    ))

    fig.add_trace(go.Scatter(
        y=[60, 65, 70, 72, 75],
        mode="lines+markers",
        name="Memory Usage"
    ))

    fig.update_layout(title="System Performance Trends")

    st.plotly_chart(fig, use_container_width=True)

    # ================= MODULE STATUS =================
    st.markdown("### 🔧 AI Modules Status")

    st.success("✔ Prediction Engine Online")
    st.success("✔ Risk Scoring Active")
    st.success("✔ Data Pipeline Running")
    st.success("✔ Analytics Engine Ready")

    # ================= ARCHITECTURE VIEW =================
    st.markdown("### 🏗 System Architecture")

    fig2 = px.sunburst(
        names=["AI System", "ML Model", "Analytics", "Reports", "UI Layer"],
        parents=["", "AI System", "AI System", "AI System", "AI System"],
        values=[100, 40, 20, 20, 20],
        title="System Structure"
    )

    st.plotly_chart(fig2, use_container_width=True)
# ---------------- FOOTER ----------------
st.markdown("---")
st.caption(f"🏥 Health Sense 360 Pro | AI Clinical System | Accuracy: {accuracy:.2%}")
