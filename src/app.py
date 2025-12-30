import os
import cv2
import numpy as np
import tensorflow as tf
import PySimpleGUI as sg
from tensorflow.keras.models import load_model

# Load pre-trained model
model_path = "brain_tumor_classification_model.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found. Ensure the model is saved correctly.")

model = load_model(model_path)

# Get expected input shape
input_shape = model.input_shape  # e.g., (None, 224, 224, 3) or (None, 128, 128, 1)
IMG_HEIGHT, IMG_WIDTH = input_shape[1], input_shape[2]
CHANNELS = input_shape[3]  # 1 for grayscale, 3 for RGB

# Define GUI Layout
sg.theme("DarkBlue")
layout = [
    [sg.Text("Brain Tumor Detection", font=("Helvetica", 16))],
    [sg.Text("Select an Image:"), sg.Input(key="-FILE-", enable_events=True), sg.FileBrowse()],
    [sg.Image(filename="", key="-IMAGE-", size=(300, 300))],
    [sg.Button("Detect Tumor", size=(12, 1)), sg.Button("Exit", size=(10, 1))],
    [sg.Text("Result: ", size=(40, 1), key="-RESULT-")]
]

# Create the window
window = sg.Window("Brain Tumor Detection", layout, finalize=True)

while True:
    event, values = window.read()
    if event in (sg.WIN_CLOSED, "Exit"):
        break
    
    if event == "-FILE-":
        img_path = values["-FILE-"]
        if os.path.exists(img_path):
            image = cv2.imread(img_path)
            image = cv2.resize(image, (300, 300))
            img_bytes = cv2.imencode(".png", image)[1].tobytes()
            window["-IMAGE-"].update(data=img_bytes)
    
    if event == "Detect Tumor":
        if values["-FILE-"]:
            # Load image in grayscale or RGB based on model requirements
            if CHANNELS == 1:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read in grayscale
            else:
                img = cv2.imread(img_path)  # Read in RGB
            
            # Resize and reshape image
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            img = img.reshape(1, IMG_HEIGHT, IMG_WIDTH, CHANNELS)  # Adjust dimensions
            img = img.astype('float32') / 255.0  # Normalize

            # Model prediction
            prediction = model.predict(img)
            result_text = "Tumor Detected" if prediction[0][0] > 0.5 else "No Tumor Detected"
            window["-RESULT-"].update(f"Result: {result_text}")

window.close()
