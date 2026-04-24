import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score, mean_absolute_error

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS (🔥 PREMIUM UI)
# -----------------------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.block-container {
    padding-top: 2rem;
}
h1 {
    font-size: 3rem;
    font-weight: 700;
}
.stButton>button {
    background: linear-gradient(90deg,#00adb5,#00f2ff);
    color: white;
    border-radius: 12px;
    height: 3.2em;
    width: 100%;
    font-size: 16px;
    font-weight: 600;
}
.stButton>button:hover {
    transform: scale(1.02);
}
.metric-box {
    padding: 20px;
    border-radius: 12px;
    background: #1c1f26;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# TITLE
# -----------------------------
st.title("🏠 AI-Powered House Price Predictor")
st.write("Predict housing prices using advanced Machine Learning with real-time insights.")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    data = fetch_california_housing(as_frame=True)
    return data.data, data.target

X, y = load_data()

# -----------------------------
# TRAIN MODEL
# -----------------------------
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return model, r2_score(y_test, y_pred), mean_absolute_error(y_test, y_pred)

model, r2, mae = train_model(X, y)

# -----------------------------
# METRICS DASHBOARD
# -----------------------------
st.subheader("📊 Model Performance")

col1, col2 = st.columns(2)
col1.metric("📈 R² Score", f"{r2:.2f}")
col2.metric("📉 Mean Absolute Error", f"{mae:.2f}")

st.divider()

# -----------------------------
# INPUT UI (🔥 MODERN)
# -----------------------------
st.subheader("🏡 Enter Property Details")

col1, col2, col3 = st.columns(3)

with col1:
    MedInc = st.slider("Median Income", 0.0, 15.0, 5.0)
    HouseAge = st.slider("House Age", 0.0, 50.0, 20.0)

with col2:
    AveRooms = st.slider("Average Rooms", 0.0, 10.0, 6.0)
    AveBedrms = st.slider("Average Bedrooms", 0.0, 5.0, 1.0)

with col3:
    Population = st.slider("Population", 0.0, 5000.0, 1000.0)
    AveOccup = st.slider("Avg Occupancy", 0.0, 10.0, 3.0)

Latitude = st.slider("Latitude", 32.0, 42.0, 34.0)
Longitude = st.slider("Longitude", -125.0, -114.0, -118.0)

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("🚀 Predict Price"):

    input_data = pd.DataFrame([[
        MedInc, HouseAge, AveRooms, AveBedrms,
        Population, AveOccup, Latitude, Longitude
    ]], columns=X.columns)

    prediction = model.predict(input_data)[0] * 100000

    # Confidence approximation
    preds = [tree.predict(input_data)[0] for tree in model.estimators_]
    confidence = np.std(preds)

    st.success(f"💰 Estimated Price: ${prediction:,.2f}")

    # -----------------------------
    # AI INSIGHTS (🔥 WOW FACTOR)
    # -----------------------------
    st.subheader("🤖 AI Insights")

    if MedInc > 6:
        st.write("✔ High income area → price increases significantly")
    if HouseAge < 10:
        st.write("✔ Newer property → higher value")
    if Latitude > 36:
        st.write("✔ Location in high-demand zone")

    st.write(f"📊 Prediction Confidence (lower is better): {confidence:.4f}")

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
st.subheader("📊 Feature Importance")

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=True)

st.bar_chart(importance_df.set_index("Feature"))

# -----------------------------
# FOOTER
# -----------------------------
st.divider()
st.markdown("""
### 🚀 About This App
- Built using **Machine Learning (Random Forest)**
- Uses real-world California housing dataset
- Designed for **portfolio & recruiters**

👨‍💻 Built by Ravi Yadav
""")