import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

# ---------------- UI ----------------
st.title("😃 Emotion Detection App (TFLite)")
st.write("Upload an image and get emotion prediction")

# ---------------- Load model ----------------
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="emotion_model_quant.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ---------------- Emotion labels ----------------
classes = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# ---------------- Upload image ----------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert image
    img = np.array(image)
    img = cv2.resize(img, (48, 48))  # adjust if your model uses different size
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img.reshape(1, 48, 48, 1).astype(np.float32) / 255.0

    # ---------------- Prediction ----------------
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    prediction = np.argmax(output)

    st.success(f"Predicted Emotion: {classes[prediction]}")
