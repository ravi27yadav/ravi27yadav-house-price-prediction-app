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
# LOAD DATA (SAFE CACHE)
# -----------------------------
@st.cache_data
def load_data():
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    return X, y

X, y = load_data()

# -----------------------------
# TRAIN MODEL (CACHE)
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
# PREDICT
# -----------------------------
if st.button("🔍 Predict Price"):
    input_data = pd.DataFrame([[
        MedInc, HouseAge, AveRooms, AveBedrms,
        Population, AveOccup, Latitude, Longitude
    ]], columns=X.columns)

    prediction = model.predict(input_data)

    st.success(f"💰 Predicted House Price: ${prediction[0]*100000:.2f}")

# -----------------------------
# FEATURE INFO
# -----------------------------
st.subheader("ℹ️ Feature Explanation")

st.markdown("""
- Median Income → Income level in the area  
- House Age → Age of houses  
- Average Rooms → Rooms per house  
- Average Bedrooms → Bedrooms per house  
- Population → Area population  
- Average Occupancy → People per house  
- Latitude & Longitude → Location  
""")

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
st.subheader("📌 Feature Importance")

importances = model.feature_importances_
df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

st.bar_chart(df.set_index("Feature"))

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("🚀 Built by Ravi Yadav")