import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing

# Title
st.title("🏠 House Price Prediction App")

# Load data
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestRegressor()
model.fit(X_train, y_train)

st.write("### Enter House Details:")

# Input fields
MedInc = st.number_input("Median Income", 0.0)
HouseAge = st.number_input("House Age", 0.0)
AveRooms = st.number_input("Average Rooms", 0.0)
AveBedrms = st.number_input("Average Bedrooms", 0.0)
Population = st.number_input("Population", 0.0)
AveOccup = st.number_input("Average Occupancy", 0.0)
Latitude = st.number_input("Latitude", 0.0)
Longitude = st.number_input("Longitude", 0.0)

# Predict button
if st.button("Predict Price"):
    input_data = pd.DataFrame([[MedInc, HouseAge, AveRooms, AveBedrms,
                                Population, AveOccup, Latitude, Longitude]],
                              columns=X.columns)

    prediction = model.predict(input_data)

    st.success(f"💰 Predicted House Price: {prediction[0]:.2f}")