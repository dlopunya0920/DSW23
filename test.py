import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the saved model
filename = "churn.sav"
loaded_model = pickle.load(open(filename, "rb"))

# Load the saved StandardScaler
scaler_filename = "scaler.sav"
loaded_scaler = pickle.load(open(scaler_filename, "rb"))

# Define the label mapping dictionaries
mapping1 = {'Jakarta': 1, 'Bandung': 2}
mapping2 = {'Mid End': 1, 'High End': 2, 'Low End': 3}
mapping3 = {'Yes': 1, 'No': 2, 'No internet service': 3}
mapping4 = {'No': 1, 'Yes': 2}
mapping5 = {'Digital Wallet': 1, 'Pulsa': 2, 'Debit': 3, 'Credit': 4}
mapping6 = {'Yes': 1, 'No': 0}

st.set_page_config(
    page_title="Churn Prediction",
    page_icon=":chart_with_upwards_trend:",  # Gantilah dengan emoji atau ikon lain yang Anda inginkan
    
)
# Streamlit app
st.title('Churn Prediction App')

# Create input fields for user input
st.sidebar.header('Input Customer Data')
tenure_months = st.sidebar.number_input('Tenure Months')
location = st.sidebar.selectbox('Location', list(mapping1.keys()))
device_class = st.sidebar.selectbox('Device Class', list(mapping2.keys()))
games_product = st.sidebar.selectbox('Games Product', list(mapping3.keys()))
music_product = st.sidebar.selectbox('Music Product', list(mapping3.keys()))
education_product = st.sidebar.selectbox('Education Product', list(mapping3.keys()))
call_center = st.sidebar.selectbox('Call Center', list(mapping4.keys()))
video_product = st.sidebar.selectbox('Video Product', list(mapping3.keys()))
use_myapp = st.sidebar.selectbox('Use MyApp', list(mapping3.keys()))
payment_method = st.sidebar.selectbox('Payment Method', list(mapping5.keys()))
monthly_purchase_thou_idr = st.sidebar.number_input('Monthly Purchase (Thou. IDR)')

# Map user input to label encoded values
location_encoded = mapping1.get(location, 0)
device_class_encoded = mapping2.get(device_class, 0)
games_product_encoded = mapping3.get(games_product, 0)
music_product_encoded = mapping3.get(music_product, 0)
education_product_encoded = mapping3.get(education_product, 0)
call_center_encoded = mapping4.get(call_center, 0)
video_product_encoded = mapping3.get(video_product, 0)
use_myapp_encoded = mapping3.get(use_myapp, 0)
payment_method_encoded = mapping5.get(payment_method, 0)

# Prepare input data for prediction
input_data = [tenure_months, location_encoded, device_class_encoded, games_product_encoded, music_product_encoded,
              education_product_encoded, call_center_encoded, video_product_encoded, use_myapp_encoded,
              payment_method_encoded, monthly_purchase_thou_idr]

# Add a "Predict" button
if st.sidebar.button('Predict'):
    input_data_array = np.array(input_data).reshape(1, -1)

    # Scale the input data using the loaded StandardScaler
    std_data = loaded_scaler.transform(input_data_array)

    # Make the prediction using the loaded model
    prediction = loaded_model.predict(std_data)

    # Display the prediction result
    st.header('Churn Prediction Result:')
    if prediction[0] == 0:
        st.write("Customer is predicted as 'Not Churn'")
    else:
        st.write("Customer is predicted as 'Churn'")
else:
    st.header('Istilah Churn:')
    st.write("Istilah churn merujuk pada konsep dalam bisnis yang mengacu pada jumlah pelanggan atau klien yang berhenti menggunakan produk atau layanan perusahaan dalam periode waktu tertentu. Churn biasanya digunakan dalam konteks layanan berlangganan, seperti langganan internet, televisi kabel, telepon seluler, platform streaming, dan bisnis lain yang melibatkan pelanggan yang membayar secara berkala.")