import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np
import pickle

import time

def loadmodel():
    model = load_model("sentiment_classifier_CNN.h5")
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
    print("[info] loading model....")
    model = loadmodel()
    print("[info] model loaded....")
    print("[info] loading tokenizer object....")
    tokenizer = loadtokens()
    print("[info] tokenizer object loaded....")
    
    while True:
        text = input("(write something)=>")
        print(predict(text,model,tokenizer))
        
    
    
main()
    
    
    
