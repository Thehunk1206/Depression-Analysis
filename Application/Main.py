import cv2
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

import numpy as np
import pandas as pd

import pickle

import time

import speech_recog
import speech_recognition as sr
from speak import TTS
# useful variables and list
classes = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']


def load_models_and_objects():
    '''loading models and object'''
    # loading CascadeClassifier
    face_cascade = cv2.CascadeClassifier()
    face_cascade.load('../../cascade/haarcascade_frontalface_alt.xml')

    # loading Emotion recognition model
    emotion_model = load_model("../assets/emotion_recogtion.h5")

    # loading tokens
    tokenizer = pickle.load(open("../assets/tokens.pkl", "rb"))

    # loading text_classifier_model
    text_model = load_model("../assets/sentiment_classifier_CNN.h5")

    return face_cascade, emotion_model, tokenizer, text_model

# =================Module 1 Emotion recognition========================
# TODO add fetch detected emotion to CSV file


def detect_emotion(videocapture, model, face_cascade):
    '''it requires capture of video, model, and instace of CascadeClassifier class'''
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
        roi_gray = grayFrame[y:y+w, x:x+h]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img = image.img_to_array(roi_gray)
        img = np.expand_dims(img, axis=0)
        img /= 255
        pred = model.predict(img)
        emotion = classes[np.argmax(pred[0])]
        print(emotion)

# ==============Module 2 Analysis of speech sentiment================


def decode_sentiment(score):
    return "NEGATIVE" if score < 0.5 else "POSITIVE"


def speech_to_sentiment(model, tokenizer):
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    text = speech_recog.recognize_speech_from_mic(recognizer, microphone)

    # Tokenize text
    if text["transciption"] != None:
        x_text = pad_sequences(tokenizer.texts_to_sequences(
            [text["transcription"]]), maxlen=300)
        # Predict
        out_put = model.predict([x_text])[0]
        #   Decode sentiment
        sentiment = decode_sentiment(out_put)
    else:
        pass

        '''
        print("You said: {}".format(text["transcription"]))
        if text["transcription"] != None:
            print(predict(text["transcription"],model,tokenizer))
        else:
            pass

        '''

# =====================prompting Quentions=============================


