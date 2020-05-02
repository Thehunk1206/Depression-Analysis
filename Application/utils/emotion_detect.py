import cv2
import tensorflow
from tensorflow.keras.preprocessing import image

import numpy as np


classes = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']


def detect_emotion(videocapture,model,face_cascade):
    ret, frame = videocapture.read()
    flipImage = cv2.flip(frame, 1)
    grayFrame = cv2.cvtColor(flipImage, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        grayFrame,
        scaleFactor=1.2,
        minNeighbors=3,
        minSize=(40, 40)
    )

    for (x, y, w, h) in faces:
        #cv2.rectangle(flipImage, (x, y), (x+w, y+h), (0, 255, 0), 1)
        roi_gray = grayFrame[y:y+w, x:x+h]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img = image.img_to_array(roi_gray)
        img = np.expand_dims(img, axis=0)
        img /= 255
        pred = model.predict(img)
        emotion = classes[np.argmax(pred[0])]
    return emotion
