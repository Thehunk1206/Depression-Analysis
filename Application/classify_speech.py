import tensorflow
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

import pandas as pd

from time import sleep
from datetime import datetime
import sys

import pickle

# utils imports
import speech_recog
import speech_recognition as sr


sentiment_data_dict = {"time": [],
                       "text": [],
                       "sentiment": []}
name = sys.argv[1]
age = sys.argv[2]

# ========================util functions============================


def load_models_and_tokens():

    # loading tokens
    tokenizer = pickle.load(open("assets/tokens.pkl", "rb"))

    # loading text_classifier_model
    text_model = load_model("assets/sentiment_classifier_CNN.h5")

    return tokenizer, text_model


def getDateTime():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def clear():
    sentiment_data_dict["time"].clear()
    sentiment_data_dict["text"].clear()
    sentiment_data_dict["sentiment"].clear()


def decode_sentiment(score):
    return "NEGATIVE" if score < 0.5 else "POSITIVE"

# ===============speech to sentiment===============================


def speech_to_sentiment(model, tokenizer):
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    try:
        while True:
            print("listening.....")
            text = speech_recog.recognize_speech_from_mic(
                recognizer, microphone)
            # Tokenize text
            if text["transcription"] != None:
                x_text = pad_sequences(tokenizer.texts_to_sequences(
                    [text["transcription"]]), maxlen=300)
                # Predict
                out_put = model.predict([x_text])[0]
                #   Decode sentiment
                sentiment = decode_sentiment(out_put)

                clear()
                sentiment_data_dict["time"].append(getDateTime())
                sentiment_data_dict["text"].append(text["transcription"])
                sentiment_data_dict["sentiment"].append(sentiment)
                df = pd.DataFrame(sentiment_data_dict)
                with open("results/Module2 "+name+" "+age+".csv", 'a') as f:
                    df.to_csv(f, header=f.tell() == 0)
                print(text["transcription"])
                print(sentiment)
                f.close()
            else:
                pass
    except KeyboardInterrupt:
        pass

# ========================main====================================


def main():
    tokenizer, text_model = load_models_and_tokens()
    speech_to_sentiment(text_model, tokenizer)


main()
