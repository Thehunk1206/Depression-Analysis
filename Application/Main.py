import cv2

import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np
import pandas as pd
import pickle
from time import sleep

import subprocess as sp

#utils imports
import speech_recog
import speech_recognition as sr
from speak import TTS

tts = TTS()

def load_models_and_tokens():

    # loading tokens
    tokenizer = pickle.load(open("assets/tokens.pkl", "rb"))

    # loading text_classifier_model
    text_model = load_model("assets/sentiment_classifier_CNN.h5")

    return tokenizer, text_model

     
# ==============Module 2 Analysis of speech sentiment================


def decode_sentiment(score):
    return "NEGATIVE" if score < 0.5 else "POSITIVE"


def speech_to_sentiment(model, tokenizer):
    print("function 2 started")
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    while True:
        print("listening.....")
        text = speech_recog.recognize_speech_from_mic(recognizer, microphone)
        print("You said: {}".format(text["transcription"]))
        # Tokenize text
        if text["transcription"] != None:
            x_text = pad_sequences(tokenizer.texts_to_sequences(
                [text["transcription"]]), maxlen=300)
            # Predict
            out_put = model.predict([x_text])[0]
            #   Decode sentiment
            sentiment = decode_sentiment(out_put)
            print(sentiment)
        else:
            pass


# =====================prompting Quentions=============================
def prompt_questionaire():
    global name
    global age

    tts.speak("Enter your name")
    name = str(input("Enter your name>> "))
    tts.speak("Enter your Age")
    age = str(input("Enter your age>> "))
    
    

#===============main fucntion========================

if __name__ == "__main__":
    #tokenizer, text_model = load_models_and_tokens()
    #p = sp.Popen(["python3","emotion_detect.py"])
    
    #speech_to_sentiment(text_model, tokenizer)
    #print("[info] Loading up........")
    #sleep(30)
    #p.terminate()

    #detect_emotion(capture,emotion_model,faces_cascade)
    prompt_questionaire()


