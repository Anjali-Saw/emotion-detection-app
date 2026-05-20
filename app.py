import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2
import numpy as np
import tensorflow as tf

st.title("Live Emotion Detection")

# Load model
interpreter = tf.lite.Interpreter(model_path="emotion_model_quant.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Emotion labels
classes = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

class EmotionDetector(VideoTransformerBase):

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        for (x, y, w, h) in faces:

            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

            face = gray[y:y+h, x:x+w]

            face = cv2.resize(face, (48, 48))

            face = face / 255.0

            face = np.reshape(face, (1, 48, 48, 1)).astype(np.float32)

            # Prediction
            interpreter.set_tensor(input_details[0]['index'], face)
            interpreter.invoke()

            prediction = interpreter.get_tensor(
                output_details[0]['index']
            )

            emotion = classes[np.argmax(prediction)]

            cv2.putText(
                img,
                emotion,
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        return img

webrtc_streamer(
    key="emotion",
    video_transformer_factory=EmotionDetector,
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    },
    media_stream_constraints={
        "video": True,
        "audio": False
    },
)
