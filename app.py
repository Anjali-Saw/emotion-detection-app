import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

st.title("😃 Live Emotion Detection (Webcam)")

# ---------------- Load model ----------------
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="emotion_model_quant.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

classes = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# ---------------- LIVE CAMERA ----------------
img_file = st.camera_input("Take a photo")

if img_file is not None:
    image = Image.open(img_file)
    st.image(image, caption="Captured Frame", use_container_width=True)

    # preprocessing
    img = np.array(image)
    img = cv2.resize(img, (48, 48))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img.reshape(1, 48, 48, 1).astype(np.float32) / 255.0

    # prediction
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    pred = np.argmax(output)

    st.success(f"Predicted Emotion: {classes[pred]}")
