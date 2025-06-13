import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

st.title("🔍 Fraud Detection ML App")

uploaded_file = st.file_uploader("📂 Upload Transaction CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Preprocessing
    df.fillna(method='ffill', inplace=True)

    # Drop unnecessary columns
    df_cleaned = df.drop(columns=[
        'x  trans_id', 'trans_date_trans_time', 'cc_num', 'first', 'last',
        'street', 'dob', 'customer_id'
    ], errors='ignore')

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

    # Predictions
    y_pred = model.predict(X_test)

    # Output results
    st.subheader("✅ Confusion Matrix")
    st.text(confusion_matrix(y_test, y_pred))

    st.subheader("✅ Classification Report")
    st.text(classification_report(y_test, y_pred))
else:
    st.warning("👆 Please upload a CSV file to proceed.")
