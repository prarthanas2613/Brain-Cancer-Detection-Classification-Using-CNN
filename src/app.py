import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained models
classification_model = load_model("brain_tumor_classification_model.h5")
detection_model = load_model("brain_tumor_detection_model.keras")

# Function to preprocess the image
def preprocess_image(image):
    img = cv2.resize(image, (240, 240))
    img = (img / 255.0).astype(np.float32)
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0) # Add batch dimension
    return img

# Streamlit app
st.title("Brain Tumor Detection and Classification")

uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make predictions
    classification_prediction = classification_model.predict(processed_image)
    detection_prediction = detection_model.predict(processed_image)

    # Display the image
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    # Display predictions
    st.write("## Predictions")
    st.write("Classification Prediction:", classification_prediction)
    st.write("Detection Prediction:", detection_prediction)
