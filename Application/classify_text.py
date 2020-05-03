from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences

import numpy as np

import time


def decode_sentiment(score):
    return "NEGATIVE" if score < 0.5 else "POSITIVE"


def predict_sentiment(text, model, tokenizer):
    # Tokenize text
    x_text = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=300)
    # Predict
    out_put = model.predict([x_text])[0]
    # Decode sentiment
    sentiment = decode_sentiment(out_put)

    return sentiment


'''
def speech_to_sentiment():
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
'''
