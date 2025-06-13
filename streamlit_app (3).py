
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import math

st.title("🔍 Machine Learning-Based Fraud Detection App")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("✅ Dataset Preview:")
    st.dataframe(df.head())

    # Drop unnecessary or problematic columns
    drop_columns = ['x  trans_id', 'trans_date_trans_time', 'cc_num', 'first', 'last', 'street', 'dob', 'customer_id']
    df_cleaned = df.drop(columns=[col for col in drop_columns if col in df.columns])

    # Fill missing values
    df_cleaned.fillna(method='ffill', inplace=True)

    # Encode categorical features
    le = LabelEncoder()
    for col in df_cleaned.select_dtypes(include='object').columns:
        df_cleaned[col] = le.fit_transform(df_cleaned[col])

    # Define features and label
    X = df_cleaned.drop("is_fraud", axis=1)
    y = df_cleaned["is_fraud"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    st.subheader("📊 Model Evaluation")
    st.text("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Geolocation Distance Verification (Optional Tool)
    st.subheader("📍 Verify Geolocation Distance")
    with st.form("geo_form"):
        registered_lat = st.number_input("Registered Latitude", value=28.6139)
        registered_lon = st.number_input("Registered Longitude", value=77.2090)
        transaction_lat = st.number_input("Transaction Latitude", value=28.7041)
        transaction_lon = st.number_input("Transaction Longitude", value=77.1025)
        threshold_km = st.number_input("Threshold Distance (km)", value=5.0)
        submitted = st.form_submit_button("Verify")

        if submitted:
            def haversine(lat1, lon1, lat2, lon2):
                lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
                c = 2 * math.asin(math.sqrt(a))
                return 6371 * c

            distance = haversine(registered_lat, registered_lon, transaction_lat, transaction_lon)
            st.write(f"📏 Distance between locations: {distance:.2f} km")
            if distance <= threshold_km:
                st.success("✅ Transaction location is within the allowed radius.")
            else:
                st.error("⚠️ Transaction location is outside the allowed radius!")
