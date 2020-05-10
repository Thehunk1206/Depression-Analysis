import pyttsx3
import time


class TTS:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.voices_id = "english"
        self.engine.setProperty('voice', self.voices_id)
        self.engine.setProperty('rate', 150)

    def speak(self, texts):
        self.engine.say(texts)
        self.engine.runAndWait()
        self.engine.stop()



