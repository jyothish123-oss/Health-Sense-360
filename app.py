import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import plotly.express as px
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Health Sense 360",
    page_icon="🏥",
    layout="wide"
)

# ---------------- SESSION STATE ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("diabetes.csv")

df = load_data()

# ---------------- TRAIN MODEL ----------------
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

# ---------------- HEADER ----------------
st.title("🏥 Health Sense 360")
st.markdown(
    "### Chronic Disease Risk Prediction & Wellness Recommendation System"
)

st.markdown("---")

# ---------------- SIDEBAR ----------------
st.sidebar.header("👤 Patient Details")

preg = st.sidebar.number_input(
    "Pregnancies", 0, 20, 1
)

glucose = st.sidebar.number_input(
    "Glucose", 0, 300, 120
)

bp = st.sidebar.number_input(
    "Blood Pressure", 0, 200, 80
)

skin = st.sidebar.number_input(
    "Skin Thickness", 0, 100, 20
)

insulin = st.sidebar.number_input(
    "Insulin", 0, 900, 80
)

bmi = st.sidebar.number_input(
    "BMI", 10.0, 60.0, 25.0
)

dpf = st.sidebar.number_input(
    "Diabetes Pedigree Function",
    0.0,
    3.0,
    0.5
)

age = st.sidebar.number_input(
    "Age",
    1,
    100,
    30
)

height = st.sidebar.number_input(
    "Height (cm)",
    100,
    220,
    170
)

weight = st.sidebar.number_input(
    "Weight (kg)",
    20,
    200,
    70
)

# ---------------- HEALTH CALCULATIONS ----------------
bmi_calc = round(
    weight / ((height / 100) ** 2),
    2
)

water_intake = round(
    weight * 0.035,
    2
)

daily_calories = round(
    (10 * weight)
    + (6.25 * height)
    - (5 * age)
    + 5
)

# ---------------- DASHBOARD CARDS ----------------
st.subheader("📊 Personal Health Dashboard")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Age", age)

with c2:
    st.metric("BMI", bmi_calc)

with c3:
    st.metric(
        "Water Intake",
        f"{water_intake} L/day"
    )

with c4:
    st.metric(
        "Calories",
        f"{daily_calories} kcal"
    )

# ---------------- BMI STATUS ----------------
if bmi_calc < 18.5:
    bmi_status = "Underweight"
elif bmi_calc < 25:
    bmi_status = "Normal"
elif bmi_calc < 30:
    bmi_status = "Overweight"
else:
    bmi_status = "Obese"

st.info(f"📌 BMI Status: {bmi_status}")

st.markdown("---")

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Health Risk"):

    input_data = np.array([[
        preg,
        glucose,
        bp,
        skin,
        insulin,
        bmi,
        dpf,
        age
    ]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(
        input_scaled
    )[0]

    probability = model.predict_proba(
        input_scaled
    )[0][1]

    risk = round(
        probability * 100,
        2
    )

    st.subheader("🩺 Diabetes Risk Assessment")

    if prediction == 1:
        st.error(
            f"⚠ High Diabetes Risk ({risk}%)"
        )
    else:
        st.success(
            f"✅ Low Diabetes Risk ({risk}%)"
        )

    # ---------------- WELLNESS SCORE ----------------
    wellness = max(
        0,
        100
        - (risk * 0.5)
        - max(0, (bmi_calc - 25) * 2)
    )

    st.subheader("🌟 Wellness Score")

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=wellness,
            title={
                "text": "Wellness Score"
            },
            gauge={
                "axis": {
                    "range": [0, 100]
                }
            }
        )
    )

    st.plotly_chart(
        gauge,
        use_container_width=True
    )

    # ---------------- RISK PIE CHART ----------------
    risk_df = pd.DataFrame({
        "Category": [
            "Risk",
            "Healthy"
        ],
        "Value": [
            risk,
            100 - risk
        ]
    })

    fig = px.pie(
        risk_df,
        values="Value",
        names="Category",
        title="Health Risk Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ---------------- RECOMMENDATIONS ----------------
    st.subheader(
        "💡 Health Recommendations"
    )

    recommendations = []

    if glucose > 140:
        recommendations.append(
            "Reduce sugar intake and monitor blood glucose."
        )

    if bmi_calc > 25:
        recommendations.append(
            "Follow a weight management plan."
        )

    if bp > 90:
        recommendations.append(
            "Reduce sodium intake."
        )

    if age > 40:
        recommendations.append(
            "Walk at least 30 minutes daily."
        )

    if len(recommendations) == 0:
        recommendations.append(
            "Maintain your healthy lifestyle."
        )

    for rec in recommendations:
        st.success(rec)

    # ---------------- DIET PLAN ----------------
    st.subheader("🍽 Personalized Diet Plan")

    if glucose > 140:
        st.write("""
        **Breakfast**
        - Oats
        - Boiled Eggs
        - Green Tea

        **Lunch**
        - Brown Rice
        - Dal
        - Salad

        **Dinner**
        - Grilled Paneer
        - Vegetables
        """)
    else:
        st.write("""
        **Breakfast**
        - Fruits
        - Milk

        **Lunch**
        - Rice
        - Vegetables

        **Dinner**
        - Chapati
        - Curry
        """)

    # ---------------- EXERCISE ----------------
    st.subheader(
        "🏃 Exercise Recommendation"
    )

    if bmi_calc > 30:
        st.warning(
            "45 minutes brisk walking daily."
        )
    elif bmi_calc > 25:
        st.info(
            "30 minutes jogging and yoga."
        )
    else:
        st.success(
            "Maintain 30 minutes of daily activity."
        )

    # ---------------- SAVE HISTORY ----------------
    st.session_state.history.append({
        "Date": datetime.now(),
        "Age": age,
        "BMI": bmi_calc,
        "Risk": risk,
        "Wellness": round(
            wellness,
            2
        )
    })

# ---------------- HISTORY ----------------
if len(st.session_state.history) > 0:

    st.markdown("---")
    st.subheader(
        "📜 Prediction History"
    )

    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        history_df,
        use_container_width=True
    )

    csv = history_df.to_csv(
        index=False
    )

    st.download_button(
        label="⬇ Download Health Report",
        data=csv,
        file_name="health_report.csv",
        mime="text/csv"
    )

    # ---------------- ANALYTICS ----------------
    if len(history_df) > 1:

        st.subheader(
            "📈 Wellness Trend"
        )

        trend = px.line(
            history_df,
            x="Date",
            y="Wellness",
            markers=True
        )

        st.plotly_chart(
            trend,
            use_container_width=True
        )

# ---------------- DATASET VIEW ----------------
with st.expander(
    "📄 View Dataset"
):
    st.dataframe(df)

# ---------------- FOOTER ----------------
st.markdown("---")

st.caption(
    f"Health Sense 360 | Model Accuracy: {accuracy:.2%}"
)