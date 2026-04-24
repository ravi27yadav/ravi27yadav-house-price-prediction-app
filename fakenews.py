import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample dataset (small for demo)
data = {
    "text": [
        "Breaking news: Market crashes due to inflation",
        "Government announces new policy reforms",
        "You won a lottery of 1 crore click here",
        "Scientists discover new species in Amazon",
        "Earn money fast with this trick click now",
        "Stock market hits record high today"
    ],
    "label": [1, 1, 0, 1, 0, 1]  # 1 = Real, 0 = Fake
}

df = pd.DataFrame(data)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2)

# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Prediction
predictions = model.predict(X_test_vec)

# Accuracy
accuracy = accuracy_score(y_test, predictions)
print("Model Accuracy:", accuracy)

# User input
print("\nEnter news text:")
user_input = input()

user_vec = vectorizer.transform([user_input])
result = model.predict(user_vec)

if result[0] == 1:
    print("Prediction: REAL NEWS")
else:
    print("Prediction: FAKE NEWS")