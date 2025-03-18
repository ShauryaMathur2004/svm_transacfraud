import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load pre-trained model and encoders
svm_model = pickle.load(open('svm_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
label_encoders = pickle.load(open('label_encoders.pkl', 'rb'))

# Load the dataset to get dynamic options for locations
data = pd.read_csv('credit_card_fraud_dataset.csv')

# Extract unique locations from the dataset
locations = data['Location'].unique().tolist()

# Convert 'TransactionDate' to datetime and extract day of the week
data['TransactionDate'] = pd.to_datetime(data['TransactionDate'])
data['TransactionDayOfWeek'] = data['TransactionDate'].dt.dayofweek
days_of_week = [0, 1, 2, 3, 4, 5, 6]  # 0 = Monday, 6 = Sunday

# Define the Streamlit app
st.title('Credit Card Fraud Detection')

st.header('Enter Transaction Details:')

# Input fields for user
transaction_id = st.number_input('Transaction ID', min_value=1, value=100001)
amount = st.number_input('Amount', min_value=0.0, value=2500.00)
merchant_id = st.number_input('Merchant ID', min_value=1, value=500)
transaction_type = st.selectbox('Transaction Type', ['purchase', 'refund'])
location = st.selectbox('Location', locations)  # Dynamic location list
transaction_hour = st.number_input('Transaction Hour (0-23)', min_value=0, max_value=23, value=14)
transaction_day_of_week = st.selectbox('Day of Week (0 = Monday, 6 = Sunday)', days_of_week)

# When the user clicks the "Predict" button
if st.button('Predict Fraud'):
    # Encode categorical variables
    encoded_transaction_type = label_encoders['TransactionType'].transform([transaction_type])[0]
    encoded_location = label_encoders['Location'].transform([location])[0]

    # Create input array
    sample = np.array([[transaction_id, amount, merchant_id, encoded_transaction_type, encoded_location, transaction_hour, transaction_day_of_week]])

    # Scale the input
    sample_scaled = scaler.transform(sample)

    # Predict using the trained model
    prediction = svm_model.predict(sample_scaled)

    # Display result
    if prediction[0] == 1:
        st.error('Warning: This transaction is predicted to be Fraudulent!')
    else:
        st.success('This transaction is predicted to be Safe (Not Fraudulent).')
