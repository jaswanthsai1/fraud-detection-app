#!/usr/bin/env python
# coding: utf-8

# In[205]:


# Fill missing values
df.fillna(method='ffill', inplace=True)

# Encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Show dataset info
print("\n✅ Dataset Info:")
print(df.info())



# In[206]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load CSV
df = pd.read_csv("Augmented_IndiaTransactMultiFacet2024.csv")

print("✅ Dataset loaded successfully.")
print(df.head())


# In[207]:


print(df.columns.tolist())


# In[208]:


# Drop unnecessary or problematic columns
df_cleaned = df.drop(columns=[
    'x  trans_id', 'trans_date_trans_time', 'cc_num', 'first', 'last',
    'street', 'dob', 'customer_id'
])

# Encode categorical features
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in df_cleaned.select_dtypes(include='object').columns:
    df_cleaned[col] = le.fit_transform(df_cleaned[col])

# ✅ Drop rows where label is NaN
df_cleaned = df_cleaned.dropna(subset=["is_fraud"])

# Define features and label
X = df_cleaned.drop("is_fraud", axis=1)
y = df_cleaned["is_fraud"]

# Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate model
from sklearn.metrics import classification_report, confusion_matrix
y_pred = model.predict(X_test)

print("\n✅ Model Evaluation:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))



# In[209]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='is_fraud', data=df)
plt.title("Fraud vs Non-Fraud Transactions")
plt.show()


# In[210]:


# Exclude non-numeric columns like timestamps
non_numeric_cols = ['Timestamp', 'Date', 'TransactionTime']  # Replace with actual column names in your data

# Optionally, print columns to find the real timestamp column name
print("🔎 Columns in dataset:", df.columns.tolist())


# In[211]:


# Geolocation Verification (Distance-Based)
import math

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    Returns distance in kilometers.
    """
    # convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))

    # Radius of earth in kilometers is 6371
    km = 6371 * c
    return km

def verify_geolocation(registered_lat, registered_lon, transaction_lat, transaction_lon, threshold_km=5):
    distance = haversine(registered_lat, registered_lon, transaction_lat, transaction_lon)
    print(f"Distance between registered and transaction location: {distance:.2f} km")

    if distance <= threshold_km:
        return True, "Transaction location is within the allowed radius."
    else:
        return False, "Transaction location is outside the allowed radius! Approval required."

# Example usage:
registered_latitude = 28.6139    # Example: New Delhi
registered_longitude = 77.2090
transaction_latitude = 28.7041   # Example: Nearby location in Delhi
transaction_longitude = 77.1025

# ✅ Correct function call on one line
is_valid, message = verify_geolocation(
    registered_latitude,
    registered_longitude,
    transaction_latitude,
    transaction_longitude,\
    threshold_km=5
)

print(is_valid, message)


# In[212]:


print(df.head())  # First 5 rows of the DataFrame
print(df.info())  # Dataset structure
print(confusion_matrix(y_test, y_pred))  # Model confusion matrix
print(classification_report(y_test, y_pred))  # Full model report


# In[213]:


# Save classification report
from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred)
with open("model_report.txt", "w") as f:
    f.write(report)

# Save DataFrame preview
df.head().to_csv("preview.csv", index=False)


# In[214]:


from IPython.display import display
display(df.head())


# In[215]:


print(df.columns.tolist())


# In[216]:


# Approval Mechanism for Out-of-Location Use
def check_location_approval(transaction, registered_location, known_locations=None):
    """
    Checks whether the transaction is made from a new or known location.
    If the location is unfamiliar, asks for user approval.

    Parameters:
        transaction (dict): The transaction details.
        registered_location (str): User's registered city/location.
        known_locations (list): Previously approved locations (optional).

    Returns:
        (bool, str): Approval status and message.
    """
    user_location = transaction['location']

    # If no known locations provided, fall back to registered location
    if known_locations is None:
        known_locations = [registered_location]

    if user_location in known_locations:
        return True, f"✅ Location '{user_location}' is known. Transaction auto-approved."
    else:
        # Simulate manual approval (here we assume user denies)
        # In real system, this would send an OTP or mobile approval prompt
        approval_needed_msg = (
            f"⚠️ Location '{user_location}' is not in user's known locations.\n"
            "📲 Requesting approval from user..."
        )
        print(approval_needed_msg)

        # Simulate user input (in production, replace with actual user interface/OTP system)
        user_input = input("Do you approve this transaction? (yes/no): ").strip().lower()
        if user_input == "yes":
            known_locations.append(user_location)  # Save as known location
            return True, f"✅ User approved new location '{user_location}'."
        else:
            return False, f"❌ Transaction denied. '{user_location}' is an unapproved location."

# 🔽 Example usage:
transaction_data = {
    'transaction_id': 1005,
    'user_id': 103,
    'amount': 1200,
    'location': 'Chennai',  # Current transaction location
    'timestamp': '2025-05-18 13:45:00'
}

registered_city = "Hyderabad"
known_cities = ["Hyderabad", "Bangalore"]

# Run approval check
is_approved, message = check_location_approval(transaction_data, registered_city, known_cities)
print("\n🔎 Approval Check Result:")
print(f"Approved: {is_approved}")
print(f"Message: {message}")


# In[217]:


#Module 5: Historical Fraud Check
import pandas as pd

# 🔹 Mock historical data created in code
historical_data = pd.DataFrame([
    {'transaction_id': 1001, 'user_id': 101, 'amount': 500, 'location': 'Hyderabad', 'is_fraud': 0},
    {'transaction_id': 1002, 'user_id': 102, 'amount': 2500, 'location': 'Delhi', 'is_fraud': 1},
    {'transaction_id': 1003, 'user_id': 103, 'amount': 700, 'location': 'Mumbai', 'is_fraud': 0},
    {'transaction_id': 1004, 'user_id': 102, 'amount': 2200, 'location': 'Delhi', 'is_fraud': 1},
    {'transaction_id': 1005, 'user_id': 104, 'amount': 1000, 'location': 'Chennai', 'is_fraud': 0},
])

def check_historical_fraud(user_id, transaction_amount, threshold_amount=1000):
    """
    Check for historical fraud for a given user using in-code mock data.
    Flags if:
    - The user had any previous fraudulent transactions.
    - The transaction amount is abnormally high (> threshold).
    """
    # Filter transactions for the given user
    user_history = historical_data[historical_data['user_id'] == user_id]

    if user_history.empty:
        return False, "✅ No past transactions found. User is new."

    # Check if any past transactions were fraudulent
    if (user_history['is_fraud'] == 1).any():
        return True, "⚠️ User has a history of fraudulent transactions."

    # Rule: High transaction amount (you can adjust this)
    if transaction_amount > threshold_amount:
        return True, f"⚠️ Transaction amount ₹{transaction_amount} exceeds safe threshold ₹{threshold_amount}."

    return False, "✅ No fraud history and transaction amount is normal."

# 🔽 Example usage
user_id_input = 102
transaction_amount_input = 1800

flagged, message = check_historical_fraud(user_id_input, transaction_amount_input)

print("Fraud Flag:", flagged)
print("Message:", message)


# In[218]:


#  Module 6: Transaction Limit Enforcement (Robust Version)

# Print all column names to help identify the correct one
print("\n Columns available for date/time and user/amount:")
print(df.columns.tolist())

# 🛠️ Set actual column names based on your dataset
date_col = 'trans_date_trans_time'      # ← Change this to your actual date column (e.g., 'trans_date')
user_col = 'customer_id'      # ← Update if user identifier is named differently
amount_col = 'amt'      # ← Make sure this is your transaction amount column

# Convert the date column to datetime format
df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

# Drop rows with invalid dates
df = df.dropna(subset=[date_col])

# Define the daily transaction limit
daily_limit = 50000

# Group by user and date to calculate total amount per day
daily_sum = df.groupby([user_col, df[date_col].dt.date])[amount_col].sum().reset_index()
daily_sum['limit_exceeded'] = daily_sum[amount_col] > daily_limit

# Show result
print("\n Transaction Limit Enforcement Summary:")
print(daily_sum['limit_exceeded'].value_counts())

# Show some examples
print("\n Transactions exceeding limit:")
print(daily_sum[daily_sum['limit_exceeded'] == True].head())


# In[219]:


 #Post-Transaction Fraud Analysis

import datetime

# Example transaction data (you can integrate with a real DB or CSV later)
completed_transactions = [
    {
        'transaction_id': 2001,
        'user_id': 101,
        'amount': 7500,
        'location': 'Delhi',
        'timestamp': '2025-05-17 02:15:00',
        'device': 'unknown_device'
    },
    {
        'transaction_id': 2002,
        'user_id': 102,
        'amount': 450,
        'location': 'Delhi',
        'timestamp': '2025-05-17 14:30:00',
        'device': 'known_device'
    }
]

def is_odd_hour(timestamp_str):
    """Check if transaction happened at an unusual time (e.g., between 1am and 5am)."""
    timestamp = datetime.datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
    return 1 <= timestamp.hour <= 5

def post_transaction_analysis(transaction):
    """
    Performs rule-based fraud checks on completed transactions.
    Returns fraud flag and reason.
    """
    flags = []

    # Rule 1: High amount
    if transaction['amount'] > 5000:
        flags.append("High transaction amount")

    # Rule 2: Odd hour transaction
    if is_odd_hour(transaction['timestamp']):
        flags.append("Transaction occurred during odd hours")

    # Rule 3: Unknown device
    if transaction['device'] == 'unknown_device':
        flags.append("Transaction made using an unrecognized device")

    # Final decision
    if flags:
        return True, f"⚠️ Fraud suspected: {' | '.join(flags)}"
    else:
        return False, "✅ Transaction appears normal"

# 🔽 Analyze all completed transactions
for tx in completed_transactions:
    flagged, message = post_transaction_analysis(tx)
    print(f"Transaction ID: {tx['transaction_id']} | Fraud Flag: {flagged} | Message: {message}")


# In[220]:


#Fraud Reporting & User Guidance
import datetime

# Simulated fraud report log
fraud_reports = []

def report_fraud(transaction, user_contact="user@example.com", country="India"):
    """
    Logs the fraud, sends simulated notification, and provides an official link to report the fraud.
    """
    # Define official reporting links by country
    cybercrime_links = {
        "India": "https://cybercrime.gov.in/",
        "USA": "https://www.ic3.gov/",
        "UK": "https://www.actionfraud.police.uk/reporting-fraud-and-cyber-crime",
        "Canada": "https://www.antifraudcentre-centreantifraude.ca/report-signalez-eng.htm"
    }

    # Get the official cybercrime report link
    report_link = cybercrime_links.get(country, "https://cybercrime.gov.in/")

    # Log fraud
    report_entry = {
        'transaction_id': transaction['transaction_id'],
        'user_id': transaction['user_id'],
        'amount': transaction['amount'],
        'location': transaction['location'],
        'timestamp': transaction['timestamp'],
        'reason': transaction.get('fraud_reason', 'Suspicious activity'),
        'reported_at': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    fraud_reports.append(report_entry)

    # Notify user (simulated)
    print("\n📢 ALERT: Fraudulent transaction detected!")
    print(f"🔍 Transaction ID: {transaction['transaction_id']} | Amount: ₹{transaction['amount']}")
    print(f"📍 Location: {transaction['location']} | Time: {transaction['timestamp']}")
    print(f"📧 Notification sent to: {user_contact}")

    print("\n🔒 Recommended Actions:")
    print("1. Block your card immediately via mobile app or helpline.")
    print("2. Contact customer support: 1800-XXX-XXXX")
    print("3. File a dispute request within 24 hours.")
    print("4. Change online banking credentials.")

    print(f"\n🌐 Report this incident to the Cyber Crime Portal:")
    print(f"🔗 {report_link} (Click to report the fraud)")

    return "✅ Fraud report logged, user alerted, and reporting link provided."

# 🔽 Example usage
suspected_transaction = {
    'transaction_id': 2001,
    'user_id': 101,
    'amount': 7500,
    'location': 'Delhi',
    'timestamp': '2025-05-17 02:15:00',
    'fraud_reason': 'Unusual time and unknown device'
}

response = report_fraud(suspected_transaction, user_contact="user101@example.com", country="India")
print(response)

