import cv2
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

import numpy as np

import pandas as pd

from datetime import datetime
from time import sleep
import sys

emotion_data_dict = {"time": [],
                     "emotions": []}

classes = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

name = sys.argv[1]
age = sys.argv[2]

# =====================util functions=============================


def load_model_objects():
    print("[ info ]Loading Cascade classifier.....")
    face_cascade = cv2.CascadeClassifier()
    isload = face_cascade.load('assets/haarcascade_frontalface_alt.xml')

    print("[ info ]Loading Model.....")
    emotion_model = load_model("assets/emotion_recogtion.h5")

    return face_cascade, emotion_model


def getDateTime():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def clear():
    emotion_data_dict["time"].clear()
    emotion_data_dict["emotions"].clear()


# =============Emotion detection module=========================
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
        emotion_data_dict["time"].append(getDateTime())
        emotion_data_dict["emotions"].append(emotion)
        df = pd.DataFrame(emotion_data_dict)
        with open("results/Module1 "+name+" "+age+".csv", 'a') as f:
            df.to_csv(f, header=f.tell() == 0)
        print(emotion)
        f.close()

# =================main================================


def main():
    cascade, emotion_model = load_model_objects()
    capture = cv2.VideoCapture(0)
    try:
        while True:
            sleep(0.7)
            _, frame = capture.read()
            detect_emotion(frame, emotion_model, cascade)
    except KeyboardInterrupt:
        capture.release()
        pass


main()
