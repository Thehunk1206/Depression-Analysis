import subprocess as sp

# utils imports
from speak import TTS
from time import sleep

tts = TTS()

# =====================prompting Quentions=============================


def getUserInfo():
    tts.speak("Enter your name")
    name = str(input("Enter your name>> "))
    tts.speak("Enter your Age")
    age = str(input("Enter your age>> "))
    return name, age


def prompt_questionaire(questionare_path):
    with open(questionare_path, 'r') as file:
        questions = file.readlines()
    for q in questions:
        print(q)
        tts.speak(q)
        resp = str(input("Press 'Enter' if finished with answer"))
        if resp == "":
            pass
        else:
            break


# ===============main fucntion========================

if __name__ == "__main__":
    name, age = getUserInfo()

    #process_detect_emotion = sp.Popen(["python3","emotion_detect.py",name,age])
    #process_speech_sentiment = sp.Popen(["python3","classify_speech.py",name,age])
    prompt_questionaire("assets/questionaire.txt")
    #process_detect_emotion.terminate()
    #process_speech_sentiment.terminate()
