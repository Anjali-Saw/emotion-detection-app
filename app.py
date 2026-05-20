import cv2
import numpy as np
import tensorflow as tf


interpreter = tf.lite.Interpreter(model_path="emotion_model_quant.tflite")
interpreter.allocate_tensors()


input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


classes = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

       
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        
        face = gray[y:y+h, x:x+w]

      
        face = cv2.resize(face, (48, 48))

        
        face = face / 255.0

        
        face = np.reshape(face, (1, 48, 48, 1)).astype(np.float32)

       
        interpreter.set_tensor(input_details[0]['index'], face)
        interpreter.invoke()

        prediction = interpreter.get_tensor(output_details[0]['index'])

        emotion = classes[np.argmax(prediction)]

       
        cv2.putText(
            frame,
            emotion,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

   
    cv2.imshow("Emotion Detection", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
