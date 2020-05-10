import cv2
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

import numpy as np

import pandas as pd

from datetime import datetime
from time import sleep
date_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

emotion_data_dict={"time":[],
                   "emotions":[]}


classes = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']


def load_model_objects():
    print("[ info ]Loading Cascade classifier.....")
    face_cascade = cv2.CascadeClassifier()
    isload = face_cascade.load('assets/haarcascade_frontalface_alt.xml')
    print(isload)

    # loading Emotion recognition model
    print("[ info ]Loading Model.....")
    emotion_model = load_model("assets/emotion_recogtion.h5")

    return face_cascade, emotion_model

def clear():
    emotion_data_dict["time"].clear()
    emotion_data_dict["emotions"].clear()


def detect_emotion(frame, model, face_cascade):
    
    flipImage = cv2.flip(frame, 1)
    grayFrame = cv2.cvtColor(flipImage, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        grayFrame,
        scaleFactor=1.2,
        minNeighbors=3,
        minSize=(40, 40)
    )

    for (x, y, w, h) in faces:
        roi_gray = grayFrame[y:y+w, x:x+h]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img = image.img_to_array(roi_gray)
        img = np.expand_dims(img, axis=0)
        img /= 255
        pred = model.predict(img)
        emotion = classes[np.argmax(pred[0])]

        clear()
        emotion_data_dict["time"].append(date_time)
        emotion_data_dict["emotions"].append(emotion)
        df = pd.DataFrame(emotion_data_dict)
        with open("emotions_capture.csv", 'a') as f:
            df.to_csv(f, header=f.tell()==0)
        f.close()


def main():
    cascade, emotion_model = load_model_objects()
    capture = cv2.VideoCapture(0)
    while True:
        _, frame = capture.read()
        detect_emotion(frame,emotion_model,cascade)
    capture.release()
main()

