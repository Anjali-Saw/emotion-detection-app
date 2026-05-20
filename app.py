import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import av
import numpy as np
import tensorflow as tf

st.title("Live Emotion Detection")

# Load TFLite model
interpreter = tf.lite.Interpreter(
    model_path="emotion_model_quant.tflite"
)

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

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

class EmotionDetector(VideoTransformerBase):

    def transform(self, frame):

        # Convert frame
        img = frame.to_ndarray(format="bgr24")

        # Resize for stability
        img = cv2.resize(img, (640, 480))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=7
        )

        for (x, y, w, h) in faces:

            try:

                # Draw rectangle
                cv2.rectangle(
                    img,
                    (x, y),
                    (x + w, y + h),
                    (255, 0, 0),
                    2
                )

                # Extract face
                face = gray[y:y+h, x:x+w]

                # Resize
                face = cv2.resize(face, (48, 48))

                # Normalize
                face = face / 255.0

                # Reshape
                face = np.reshape(
                    face,
                    (1, 48, 48, 1)
                ).astype(np.float32)

                # Prediction
                interpreter.set_tensor(
                    input_details[0]['index'],
                    face
                )

                interpreter.invoke()

                prediction = interpreter.get_tensor(
                    output_details[0]['index']
                )

                emotion = classes[np.argmax(prediction)]

                # Show emotion
                cv2.putText(
                    img,
                    emotion,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

            except Exception as e:
                print(e)

        return img

# WebRTC Stream
webrtc_streamer(
    key="emotion",
    video_transformer_factory=EmotionDetector,
    async_processing=True,
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
        ]
    },
    media_stream_constraints={
        "video": True,
        "audio": False
    },
)
