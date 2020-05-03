import tensorflow 
from tensorflow.keras.models import load_model
import pickle

import cv2

# load text and emotion model
def load_text_model():
    text_model = load_model("../assets/sentiment_classifier_CNN.h5")
    return text_model


def load_emotion_model():
    model = load_model("../assets/emotion_recogtion.h5")
    return model

#load token object
def loadtokens():
    tokenizer = pickle.load(open("../assets/tokens.pkl", "rb"))
    return tokenizer

#load cascade classifier
def load_faceCascade():
    face_cascade = cv2.CascadeClassifier()
    face_cascade.load('../../cascade/haarcascade_frontalface_alt.xml')
    return face_cascade
