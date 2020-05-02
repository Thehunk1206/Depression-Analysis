import pyttsx3
import time

class TTS:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice',self.voices[0].id)
        self.engine.setProperty('rate',150)

    def speak(self,texts):
        self.engine.say(texts)
        self.engine.runAndWait()
        self.engine.stop()


tts = TTS()
TTS.speak("hello")