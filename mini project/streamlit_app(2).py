import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

st.title("🔍 Fraud Detection using Machine Learning")
st.write("Upload your transaction dataset (.csv) to detect potential fraudulent activity.")

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded and loaded successfully!")

        drop_cols = ['x  trans_id', 'trans_date_trans_time', 'cc_num', 'first', 'last',
                     'street', 'dob', 'customer_id']
        df.drop(columns=drop_cols, inplace=True, errors='ignore')
        df.fillna(method='ffill', inplace=True)

        le = LabelEncoder()
        for col in df.select_dtypes(include='object').columns:
            df[col] = le.fit_transform(df[col])

        X = df.drop("is_fraud", axis=1)
        y = df["is_fraud"]

        if y.isnull().sum() > 0:
            st.error("🚫 Target column 'is_fraud' contains missing values.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.subheader("✅ Model Evaluation")
            st.text("Confusion Matrix:")
            st.write(confusion_matrix(y_test, y_pred))
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
else:
    st.info("👆 Please upload a CSV file to proceed.")
