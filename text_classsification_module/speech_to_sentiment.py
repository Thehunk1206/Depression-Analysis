import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

import speech_recognition as sr
import speechRecognition

import numpy as np
import pickle

import time

def loadmodel():
    model = load_model("utils/sentiment_classifier_CNN.h5")
    return model

def loadtokens():
    tokenizer = pickle.load(open("utils/tokens.pkl", "rb"))
    return tokenizer


def decode_sentiment(score):
  return "NEGATIVE" if score < 0.5 else "POSITIVE"

def predict(text,model,tokenizer):
    start_at = time.time()
    # Tokenize text
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=300)
    # Predict
    out_put = model.predict([x_test])[0]
    # Decode sentiment
    label = decode_sentiment(out_put)

    return {"label": label, "score": float(out_put),"time_taken": float((time.time())-start_at)}  

def main():
    
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    
    print("[info] loading model....")
    model = loadmodel()
    print("[info] model loaded....")
    print("[info] loading tokenizer object....")
    tokenizer = loadtokens()
    print("[info] tokenizer object loaded....")
    
    while True:
        print("listening.....")
        text = speechRecognition.recognize_speech_from_mic(recognizer, microphone)
        print("You said: {}".format(text["transcription"]))
        if text["transcription"] != None:
            print(predict(text["transcription"],model,tokenizer))
        else:
            pass
        
main()
    
    
    
