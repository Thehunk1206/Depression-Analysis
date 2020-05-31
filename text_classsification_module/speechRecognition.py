import random
import time

import speech_recognition as sr


def recognize_speech_from_mic(recognizer, microphone):

    # check that recognizer and microphone arguments are appropriate type
    if not isinstance(recognizer, sr.Recognizer):
        raise TypeError("`recognizer` must be `Recognizer` instance")

    if not isinstance(microphone, sr.Microphone):
        raise TypeError("`microphone` must be `Microphone` instance")

    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        
    response = {
        "success": True,
        "error": None,
        "transcription": None
    }

    try:
        response["transcription"] = recognizer.recognize_google(audio)
    except sr.RequestError:
        response["success"] = False
        response["error"] = "API unavailable"
    except sr.UnknownValueError:
        response["error"] = "Unable to recognize speech"

    return response

'''
if __name__ == "__main__":
    
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    
    
    while True:
        print("listening.....")
        guess = recognize_speech_from_mic(recognizer, microphone)
        print(guess["success"])
        print("You said: {}".format(guess["transcription"]))

'''
