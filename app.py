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
st.set_page_config(page_title="House Price Prediction", page_icon="🏠")

# -----------------------------
# TITLE
# -----------------------------
st.title("🏠 House Price Prediction App")

st.markdown("""
### 📊 Predict California Housing Prices using Machine Learning

Enter house features below and get an instant prediction.
""")

# -----------------------------
# LOAD DATA (SAFE)
# -----------------------------
@st.cache_data
def load_data():
    data = fetch_california_housing(as_frame=True)
    return data.data, data.target

X, y = load_data()

# -----------------------------
# TRAIN MODEL (CACHED)
# -----------------------------
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    return model, r2, mae

model, r2, mae = train_model(X, y)

# -----------------------------
# MODEL PERFORMANCE
# -----------------------------
st.subheader("📊 Model Performance")
st.write(f"R² Score: **{r2:.2f}**")
st.write(f"Mean Absolute Error: **{mae:.2f}**")

# -----------------------------
# INPUT
# -----------------------------
st.subheader("🏡 Enter House Details")

MedInc = st.number_input("Median Income", 0.0, value=5.0)
HouseAge = st.number_input("House Age", 0.0, value=20.0)
AveRooms = st.number_input("Average Rooms", 0.0, value=6.0)
AveBedrms = st.number_input("Average Bedrooms", 0.0, value=1.0)
Population = st.number_input("Population", 0.0, value=1000.0)
AveOccup = st.number_input("Average Occupancy", 0.0, value=3.0)
Latitude = st.number_input("Latitude", value=34.0)
Longitude = st.number_input("Longitude", value=-118.0)

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("🔍 Predict Price"):
    input_data = pd.DataFrame([[
        MedInc, HouseAge, AveRooms, AveBedrms,
        Population, AveOccup, Latitude, Longitude
    ]], columns=X.columns)

    prediction = model.predict(input_data)

    st.success(f"💰 Predicted House Price: ${prediction[0]*100000:.2f}")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("🚀 Built by Ravi Yadav")