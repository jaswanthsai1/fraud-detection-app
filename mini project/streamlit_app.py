import streamlit as st
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import random
import requests

# ------------------------------
# Function to calculate distance using Haversine formula
# ------------------------------
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return 6371 * c

# ------------------------------
# Function to verify geolocation
# ------------------------------
def verify_geolocation(reg_lat, reg_lon, trans_lat, trans_lon, threshold_km=5):
    distance = haversine(reg_lat, reg_lon, trans_lat, trans_lon)
    if distance <= threshold_km:
        return True, f"✅ Distance: {distance:.2f} km - Within threshold"
    else:
        return False, f"🚨 Distance: {distance:.2f} km - Outside threshold"

# ------------------------------
# Function to simulate OTP check
# ------------------------------
def simulate_otp():
    otp = random.randint(100000, 999999)
    st.info(f"Simulated OTP sent: {otp}")
    user_otp = st.text_input("Enter the OTP you received:", max_chars=6)
    if st.button("Verify OTP"):
        if user_otp == str(otp):
            st.success("✅ OTP Verified Successfully!")
        else:
            st.error("❌ Invalid OTP. Transaction flagged!")

# ------------------------------
# Function to check URL safety (mocked)
# ------------------------------
def check_url_fraud(url):
    # Simulated phishing check (replace with real API call)
    suspicious_keywords = ["login", "verify", "bank", "update", "confirm"]
    if any(keyword in url.lower() for keyword in suspicious_keywords):
        return "⚠️ Suspicious link detected! Might be phishing."
    return "✅ Link appears safe."

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("💳 Fraud Detection using Machine Learning")
st.write("Upload your transaction CSV file to detect potential frauds.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")

        # Drop non-useful or high-missing columns
        columns_to_drop = [
            'x  trans_id', 'trans_date_trans_time', 'cc_num', 'first', 'last',
            'street', 'dob', 'customer_id'
        ]
        df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True, errors='ignore')

        # Handle missing values
        df.fillna(method='ffill', inplace=True)
        df.dropna(inplace=True)

        # Encode categorical columns
        le = LabelEncoder()
        for col in df.select_dtypes(include='object').columns:
            df[col] = le.fit_transform(df[col])

        # Define features and target
        if 'is_fraud' not in df.columns:
            st.error("🚫 'is_fraud' column missing in the dataset!")
        else:
            X = df.drop("is_fraud", axis=1)
            y = df["is_fraud"]

            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train Random Forest
            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            # Predict and evaluate
            y_pred = model.predict(X_test)

            st.subheader("📊 Model Evaluation")
            st.text("Confusion Matrix:")
            st.write(confusion_matrix(y_test, y_pred))
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

            # Geolocation validation
            st.subheader("📍 Test Geolocation Verification")
            reg_lat = st.number_input("Registered Latitude", value=28.6139)
            reg_lon = st.number_input("Registered Longitude", value=77.2090)
            trans_lat = st.number_input("Transaction Latitude", value=28.7041)
            trans_lon = st.number_input("Transaction Longitude", value=77.1025)
            if st.button("Verify Location"):
                valid, msg = verify_geolocation(reg_lat, reg_lon, trans_lat, trans_lon)
                st.write(msg)

            # OTP Simulation
            st.subheader("🔐 Secondary OTP Verification")
            simulate_otp()

            # URL Checker
            st.subheader("🔗 Link Fraud Check")
            user_link = st.text_input("Paste a suspicious link here:")
            if st.button("Check Link") and user_link:
                result = check_url_fraud(user_link)
                st.warning(result)
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
else:
    st.info("⬆️ Upload a CSV file to begin.")
