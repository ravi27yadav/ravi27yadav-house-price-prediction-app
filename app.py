import streamlit as st
import pandas as pd
import numpy as np

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Real Estate India",
    page_icon="🏠",
    layout="wide"
)

# -----------------------------
# STYLE
# -----------------------------
st.markdown("""
<style>
body {background-color: #0e1117;}
.stButton>button {
    background: linear-gradient(90deg,#ff7e5f,#feb47b);
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# TITLE
# -----------------------------
st.title("🇮🇳 AI Real Estate Advisor")
st.write("Find, predict & analyze house prices across India with AI.")

# -----------------------------
# INPUT
# -----------------------------
st.subheader("🏡 Enter Property Details")

col1, col2, col3 = st.columns(3)

with col1:
    city = st.selectbox("City", ["Bangalore", "Mumbai", "Delhi", "Hyderabad"])
    size = st.slider("House Size (sqft)", 500, 5000, 1200)

with col2:
    bedrooms = st.slider("Bedrooms", 1, 5, 2)
    age = st.slider("Property Age", 0, 30, 5)

with col3:
    location_score = st.slider("Location Score", 1, 10, 7)
    amenities = st.slider("Amenities Score", 1, 10, 6)

# -----------------------------
# SIMPLE AI MODEL (SIMULATED)
# -----------------------------
base_price = {
    "Bangalore": 6000,
    "Mumbai": 15000,
    "Delhi": 8000,
    "Hyderabad": 5000
}

price = (
    size * base_price[city]
    + bedrooms * 200000
    - age * 50000
    + location_score * 100000
    + amenities * 80000
)

# -----------------------------
# PREDICT BUTTON
# -----------------------------
if st.button("🚀 Predict Price"):
    st.success(f"💰 Estimated Price: ₹{price:,.0f}")

    # -----------------------------
    # AI INSIGHTS
    # -----------------------------
    st.subheader("🤖 AI Insights")

    if location_score > 8:
        st.write("✔ Prime location → high investment potential")

    if age < 5:
        st.write("✔ New property → higher resale value")

    if city == "Mumbai":
        st.write("⚠ Mumbai market is premium → prices inflated")

    # -----------------------------
    # INVESTMENT ADVICE
    # -----------------------------
    st.subheader("📊 Investment Suggestion")

    if price < 8000000:
        st.success("💡 Good investment opportunity")
    else:
        st.warning("⚠ Slightly overpriced compared to market")

# -----------------------------
# PROPERTY LISTINGS (🔥 REAL APP FEEL)
# -----------------------------
st.subheader("🏘️ Similar Properties")

properties = [
    {"name": "2BHK in Bangalore", "price": "₹75 Lakhs", "contact": "Rahul - 9876543210"},
    {"name": "3BHK in Mumbai", "price": "₹2.1 Cr", "contact": "Amit - 9123456780"},
    {"name": "2BHK in Delhi", "price": "₹90 Lakhs", "contact": "Neha - 9988776655"},
]

cols = st.columns(3)

for i, prop in enumerate(properties):
    with cols[i]:
        st.markdown(f"""
        ### {prop['name']}
        💰 {prop['price']}  
        📞 {prop['contact']}
        """)

# -----------------------------
# AI CHAT ASSISTANT (🔥 WOW FEATURE)
# -----------------------------
st.subheader("💬 AI Assistant")

user_query = st.text_input("Ask anything about property...")

if user_query:
    if "investment" in user_query.lower():
        st.write("AI: Bangalore & Hyderabad are best for growth 📈")
    elif "cheap" in user_query.lower():
        st.write("AI: Try outskirts areas for lower prices 💰")
    else:
        st.write("AI: This property looks good based on current trends 👍")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("🚀 Built by Ravi Yadav | AI Real Estate App")