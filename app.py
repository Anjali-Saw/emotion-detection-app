import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("final_emotion_model.keras")

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

# Load Haarcascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Streamlit UI
st.title("Real-Time Emotion Detection")

run = st.checkbox("Start Webcam")

FRAME_WINDOW = st.image([])

# Webcam
cap = cv2.VideoCapture(0)

while run:

    ret, frame = cap.read()

    if not ret:
        st.write("Failed to access webcam")
        break

    # Flip webcam
    frame = cv2.flip(frame, 1)

    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.3,
        minNeighbors=5
    )

    # Process faces
    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        face = gray_frame[y:y+h, x:x+w]

        # Improve contrast
        face = cv2.equalizeHist(face)

        img = cv2.resize(face, (48, 48))

        img = img / 255.0

        img = np.reshape(img, (1, 48, 48, 1))

        # Prediction
        prediction = model.predict(img, verbose=0)

        class_index = np.argmax(prediction)

        confidence = np.max(prediction)

        emotion = classes[class_index]

        # Display prediction
        cv2.putText(
            frame,
            f"{emotion} ({confidence*100:.1f}%)",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # Convert BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Show frame in Streamlit
    FRAME_WINDOW.image(frame)

cap.release()