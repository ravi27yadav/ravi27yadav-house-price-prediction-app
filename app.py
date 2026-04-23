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
# TITLE & DESCRIPTION
# -----------------------------
st.title("🏠 House Price Prediction App")

st.markdown("""
### 📊 Predict California Housing Prices using Machine Learning

Enter house features below and get an instant prediction.

This app uses a **Random Forest Regressor** trained on real housing data.
""")

# -----------------------------
# LOAD DATA
# -----------------------------
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# -----------------------------
# TRAIN MODEL
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

# -----------------------------
# MODEL PERFORMANCE
# -----------------------------
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

st.subheader("📊 Model Performance")
st.write(f"R² Score: **{r2:.2f}**")
st.write(f"Mean Absolute Error: **{mae:.2f}**")

# -----------------------------
# FEATURE INPUT
# -----------------------------
st.subheader("🏡 Enter House Details")

MedInc = st.number_input("Median Income", min_value=0.0, value=5.0)
HouseAge = st.number_input("House Age", min_value=0.0, value=20.0)
AveRooms = st.number_input("Average Rooms", min_value=0.0, value=6.0)
AveBedrms = st.number_input("Average Bedrooms", min_value=0.0, value=1.0)
Population = st.number_input("Population", min_value=0.0, value=1000.0)
AveOccup = st.number_input("Average Occupancy", min_value=0.0, value=3.0)
Latitude = st.number_input("Latitude", value=34.0)
Longitude = st.number_input("Longitude", value=-118.0)

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("🔍 Predict Price"):
    input_data = pd.DataFrame([[MedInc, HouseAge, AveRooms, AveBedrms,
                                Population, AveOccup, Latitude, Longitude]],
                              columns=X.columns)

    prediction = model.predict(input_data)

    st.success(f"💰 Predicted House Price: ${prediction[0]*100000:.2f}")

# -----------------------------
# FEATURE EXPLANATION
# -----------------------------
st.subheader("ℹ️ Feature Explanation")

st.markdown("""
- **Median Income** → Income level in the area  
- **House Age** → Age of houses  
- **Average Rooms** → Rooms per house  
- **Average Bedrooms** → Bedrooms per house  
- **Population** → Total population in area  
- **Average Occupancy** → People per house  
- **Latitude & Longitude** → Location coordinates  
""")

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
st.subheader("📌 Feature Importance")

importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

st.bar_chart(feature_importance_df.set_index("Feature"))

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("🚀 Built by Ravi Yadav | Aspiring Full Stack & ML Developer")