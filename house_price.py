import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.datasets import fetch_california_housing

print("=== House Price Prediction Model ===\n")

# Load dataset
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model (UPGRADED)
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Metrics
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print("Model Performance:")
print(f"R2 Score: {r2:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")

# Sample prediction
sample = X_test.iloc[0:1]
predicted_price = model.predict(sample)

print("\nSample Prediction:")
print(f"Predicted Price: {predicted_price[0]:.2f}")

# Visualization
plt.scatter(y_test, predictions)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()

# -------------------------
# FIXED USER INPUT SECTION
# -------------------------

print("\nEnter house features to predict price:")
print("Feature order:")
print(list(X.columns))

print("\nExample input:")
print("8.3252 41 6.984127 1.02381 322 2.555556 37.88 -122.23")

try:
    values = [float(x) for x in input("\nEnter values: ").split()]

    if len(values) != len(X.columns):
        print(f"❌ Please enter exactly {len(X.columns)} values.")
    else:
        # Convert to DataFrame with correct column names
        user_df = pd.DataFrame([values], columns=X.columns)

        result = model.predict(user_df)
        print(f"✅ Predicted Price: {result[0]:.2f}")

except:
    print("❌ Invalid input. Please enter numeric values only.")