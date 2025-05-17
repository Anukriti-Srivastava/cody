
# main.py

import os
import sys
import threading
import datetime
import webbrowser
import pyttsx3
import speech_recognition as sr
import tkinter as tk

from gui import AssistantGUI
from file_manager import FileManager
from browser_action import COMMAND_MAP as BROWSER_COMMANDS

# Constants
WAKE_WORD = "hello"
MAX_LISTEN_TIMEOUT = 10
PHRASE_TIME_LIMIT = 15
LANGUAGE_CODE = 'en-IN'


def initialize_engine():
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[1].id if len(voices) > 1 else voices[0].id)
        engine.setProperty('rate', 150)
        return engine
    except Exception as e:
        print(f"TTS init error: {e}")
        return None


def speak(text):
    engine = initialize_engine()
    if engine:
        engine.say(text)
        engine.runAndWait()
    else:
        print(text)


import speech_recognition as sr

def listen():
    r = sr.Recognizer()
    r.energy_threshold = 5  # set a fixed energy threshold
    r.pause_threshold = 1

    with sr.Microphone(sample_rate=16000, chunk_size=1024) as source:
        r.adjust_for_ambient_noise(source, duration=2)
        try:
            audio = r.listen(
                source,
                timeout=MAX_LISTEN_TIMEOUT,
                phrase_time_limit=PHRASE_TIME_LIMIT
            )
            return r.recognize_google(audio, language=LANGUAGE_CODE).lower()
        except sr.WaitTimeoutError:
            speak("I didn't hear anything—please try again.")
        except sr.UnknownValueError:
            speak("Sorry, I couldn't understand you.")
        except sr.RequestError as e:
            speak(f"Speech service error: {e}")
        return None



class Assistant:
    def __init__(self, gui):
        self.gui = gui
        self.fm = FileManager()
        # Load any legacy intents if needed
        try:
            import json
            with open("intents.json") as f:
                self.intents = json.load(f)
        except Exception:
            self.intents = {}

    def run(self):
        speak("Cody is online. Say 'Hello' to wake me up.")
        while True:
            text = listen()
            if text and WAKE_WORD in text:
                self.gui.update_status("Listening...")
                self.gui.start_listening_animation()
                speak("How can I help?")
                cmd = listen() or ""
                self.gui.update_status(f"Command: {cmd}")
                self.handle(cmd)
                self.gui.update_status("Say 'Hello' to wake me up...")
                self.gui.stop_listening_animation()

    def handle(self, cmd):
        lowered = cmd.strip().lower()

        # 1) File open commands
        if lowered.startswith('open '):
            filename = lowered.replace('open ', '').strip()
            paths = self.fm.find_paths(filename)
            if paths:
                speak(f"Opening {os.path.basename(paths[0])}")
                self.fm.open(paths[0])
            else:
                speak("File not found.")
            return

        # 2) Legacy time/date/search
        if 'time' in lowered:
            speak(datetime.datetime.now().strftime("%I:%M %p"))
            return
        if 'day' in lowered:
            speak(datetime.datetime.today().strftime("%A"))
            return
        if lowered.startswith('search for'):
            q = lowered.replace('search for', '').strip()
            speak(f"Searching for {q}")
            webbrowser.open(f"https://google.com/search?q={q}")
            return

        # 3) Browser action commands
        for trigger, action in BROWSER_COMMANDS.items():
            if lowered.startswith(trigger):
                action(cmd)
                return

        # 4) Exit
        if lowered in ('exit', 'quit'):
            speak("Goodbye")
            sys.exit(0)

        # 5) Fallback
        speak("I didn't understand that.")


if __name__ == "__main__":
    root = tk.Tk()
    gui = AssistantGUI(root)
    assistant = Assistant(gui)
    threading.Thread(target=assistant.run, daemon=True).start()
    root.mainloop()

