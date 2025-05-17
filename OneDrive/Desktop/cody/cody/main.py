# main.py

import os
import sys
import threading
import datetime
import webbrowser
import pyttsx3
import speech_recognition as sr
import tkinter as tk
import random
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import requests # For weather API
import re # For basic entity extraction

# Assuming these files exist from your description
from gui import AssistantGUI
from file_manager import FileManager
# from browser_action import COMMAND_MAP as BROWSER_COMMANDS # Keep if needed

# --- Constants ---
WAKE_WORD = "hello" # Consider making this more unique like "hey cody"
MAX_LISTEN_TIMEOUT = 7
PHRASE_TIME_LIMIT = 10
LANGUAGE_CODE = 'en-IN' # Or 'en-US'
MODEL_DIR = 'saved_models'
INTENTS_FILE = 'intents.json'
MODEL_PATH = os.path.join(MODEL_DIR, 'intent_model.h5')
TOKENIZER_PATH = os.path.join(MODEL_DIR, 'tokenizer.pkl')
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')
MAX_LEN = 25 # Should match the 'max_len' used during training
CONFIDENCE_THRESHOLD = 0.70 # Adjusted threshold - tune based on testing
WEATHER_API_KEY = "YOUR_OPENWEATHERMAP_API_KEY" # <<<--- IMPORTANT: PASTE YOUR API KEY HERE
WEATHER_BASE_URL = "http://api.openweathermap.org/data/2.5/weather?"

# --- Global Engine ---
engine = None

def initialize_engine():
    global engine
    if engine is None:
        try:
            engine = pyttsx3.init()
            # Optional: Set voice, rate etc.
            print("TTS Engine Initialized.")
        except Exception as e:
            print(f"FATAL: TTS initialization error: {e}")
            engine = None
    return engine

def speak(text):
    global engine
    print(f"Assistant: {text}") # Always print what is being spoken
    if not engine:
        print("TTS Engine not available.")
        return
    try:
        # Clear queue before speaking new phrase (prevents overlapping)
        engine.stop()
        engine.say(text)
        engine.runAndWait()
    except RuntimeError as e:
        # Handle potential engine busy errors if runAndWait is interrupted
         print(f"TTS Runtime error: {e}. Attempting to continue.")
         # Re-initialize engine might be needed in some cases, but start simple
         # initialize_engine()
    except Exception as e:
        print(f"Error during speech: {e}")

def listen():
    r = sr.Recognizer()
    r.pause_threshold = 0.8

    with sr.Microphone(sample_rate=16000) as source:
        print("Adjusting for ambient noise...")
        try:
            r.adjust_for_ambient_noise(source, duration=1.5)
            print(f"Ambient noise adjustment complete. Threshold: {r.energy_threshold}")
            print("Listening...")
            audio = r.listen(
                source,
                timeout=MAX_LISTEN_TIMEOUT,
                phrase_time_limit=PHRASE_TIME_LIMIT
            )
            print("Audio captured. Recognizing...")
            recognized_text = r.recognize_google(audio, language=LANGUAGE_CODE).lower()
            print(f"Recognized: {recognized_text}")
            return recognized_text
        except sr.WaitTimeoutError:
            print("Listening timeout.")
            return None
        except sr.UnknownValueError:
            # Don't speak error here, let the calling function decide
            print("Recognition failed (Unknown Value).")
            return None
        except sr.RequestError as e:
            speak("Sorry, my speech service is unavailable right now.")
            print(f"Speech service error: {e}")
            return None
        except Exception as e:
            # Don't speak generic error here
            print(f"Unexpected listening error: {e}")
            return None

class Assistant:
    def __init__(self, gui):
        self.gui = gui
        self.fm = FileManager()
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.intents_data = None

        self._load_model_artifacts()
        self._load_intents()

    def _load_model_artifacts(self):
        if not all([os.path.exists(p) for p in [MODEL_PATH, TOKENIZER_PATH, LABEL_ENCODER_PATH]]):
             print("------------------------------------------------------")
             print("ERROR: Model artifacts not found!")
             print(f"Ensure these files exist in the '{MODEL_DIR}' directory:")
             print(f" - {os.path.basename(MODEL_PATH)}")
             print(f" - {os.path.basename(TOKENIZER_PATH)}")
             print(f" - {os.path.basename(LABEL_ENCODER_PATH)}")
             print("Please train the model first using train_model.py")
             print("------------------------------------------------------")
             speak("Critical error: I couldn't load my brain. Please check the model files and train me if necessary.")
             # Optionally exit or disable intent prediction
             # sys.exit(1)
             self.model = None
             return
        try:
            # Load model with custom objects if SelfAttention layer is used directly
            # from train_model import SelfAttention # Import if needed
            # custom_objects = {'SelfAttention': SelfAttention}
            # self.model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
            # If SelfAttention is not needed as custom object (e.g. if using built-in layers):
            self.model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            print("Intent recognition model loaded successfully.")

            with open(TOKENIZER_PATH, 'rb') as handle:
                self.tokenizer = pickle.load(handle)
            print("Tokenizer loaded successfully.")

            with open(LABEL_ENCODER_PATH, 'rb') as handle:
                self.label_encoder = pickle.load(handle)
            print("Label encoder loaded successfully.")
            print(f"Loaded classes: {self.label_encoder.classes_}")


        except Exception as e:
            print(f"An unexpected error occurred loading model artifacts: {e}")
            speak("An unexpected error occurred while loading my core functions.")
            self.model = None

    def _load_intents(self):
        try:
            with open(INTENTS_FILE, 'r') as f:
                self.intents_data = json.load(f)
            print("Intents data loaded successfully.")
        except FileNotFoundError:
            print(f"Error: Intents file not found at {INTENTS_FILE}")
            speak("I can't load my response patterns. Please check the intents file.")
            self.intents_data = {"intents": []}
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {INTENTS_FILE}")
            speak("My response patterns seem to be corrupted.")
            self.intents_data = {"intents": []}
        except Exception as e:
            print(f"An unexpected error occurred loading intents: {e}")
            self.intents_data = {"intents": []}

    def predict_intent(self, text):
        if not self.model or not self.tokenizer or not self.label_encoder:
            print("Prediction unavailable: Model or artifacts not loaded.")
            return None, 0.0

        try:
            # Preprocess: Remove placeholders before prediction? Or handle in patterns.
            # For now, predict on raw text lowercased.
            clean_text = text.lower().strip()
            sequences = self.tokenizer.texts_to_sequences([clean_text])
            padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

            prediction = self.model.predict(padded_sequences, verbose=0)
            predicted_index = np.argmax(prediction)
            confidence = float(np.max(prediction)) # Ensure it's a standard float

            # Decode
            if predicted_index < len(self.label_encoder.classes_):
                 predicted_tag = self.label_encoder.inverse_transform([predicted_index])[0]
            else:
                 print(f"Error: Predicted index {predicted_index} out of bounds for label encoder classes ({len(self.label_encoder.classes_)}).")
                 return "unknown", 0.0 # Fallback tag

            print(f"Predicted tag: {predicted_tag}, Confidence: {confidence:.4f}")
            return predicted_tag, confidence
        except Exception as e:
            print(f"Error during intent prediction: {e}")
            # Reraise or handle more gracefully? For now, return unknown.
            return "unknown", 0.0

    def get_response(self, tag, **kwargs):
        if not self.intents_data:
            return "I seem to have lost my ability to respond."

        for intent in self.intents_data.get('intents', []): # Safer access
            if intent.get('tag') == tag:
                response = random.choice(intent.get('responses', ["..."])) # Safer access
                try:
                    # Fill known placeholders
                    kwargs.setdefault('current_time', datetime.datetime.now().strftime('%I:%M %p'))
                    kwargs.setdefault('current_date', datetime.datetime.now().strftime('%A, %B %d, %Y'))
                    # Add defaults for weather placeholders to avoid errors if API fails
                    kwargs.setdefault('temperature', 'N/A')
                    kwargs.setdefault('conditions', 'N/A')
                    kwargs.setdefault('weather_info', 'Weather data unavailable.')
                    kwargs.setdefault('search_query', 'your topic') # Default for search
                    kwargs.setdefault('filename', 'the file') # Default for file open

                    return response.format(**kwargs)
                except KeyError as e:
                    print(f"Warning: Missing key {e} for formatting response for tag '{tag}'. Response: '{response}'")
                    # Return the raw response if formatting fails for unexpected keys
                    return response
        # If tag not found in intents data
        print(f"Warning: Tag '{tag}' not found in intents data file.")
        return "I recognized that category, but I don't have a specific response pattern for it right now."

    def run(self):
        if not initialize_engine():
            print("Exiting due to TTS engine failure.")
            self.gui.update_status("ERROR: TTS Failed")
            return

        # Check if model loaded before starting loop
        if not self.model:
             speak("Assistant functionality limited: Intent recognition model not loaded.")
             self.gui.update_status("WARN: Model Error")
        else:
             speak("Cody is online. Say 'Hello' or 'Hey Cody' to wake me up.")
             self.gui.update_status("Idle | Say 'Hello'...")


        while True:
            wake_input = listen()

            if wake_input and (WAKE_WORD in wake_input or "hey cody" in wake_input):
                self.gui.update_status("Awake | Listening...")
                self.gui.start_listening_animation()
                speak(random.choice(["Yes?", "How can I help?", "I'm listening.", "What's up?"]))

                cmd = listen()
                self.gui.stop_listening_animation() # Stop animation after listening attempt

                if cmd:
                    self.gui.update_status(f"Processing: {cmd[:30]}...") # Show truncated command
                    self.handle(cmd)
                else:
                    # Failed to hear command after wake word
                    speak(random.choice(["Sorry, I didn't catch that.", "Did you say something?", "Hmm?"]))
                    self.gui.update_status("No command heard.")

                # Reset state
                self.gui.update_status("Idle | Say 'Hello'...")
            else:
                # Optional delay if needed
                # import time
                # time.sleep(0.1)
                pass # Continue listening for wake word

    def handle(self, command):
        if not command:
            # Should not happen if called from run, but as safeguard
            speak("An empty command was received.")
            return

        lowered_cmd = command.strip().lower()
        predicted_tag, confidence = self.predict_intent(lowered_cmd)
        response_kwargs = {} # For placeholder values

        # --- Intent Handling (Primary) ---
        if predicted_tag and predicted_tag != "unknown" and confidence >= CONFIDENCE_THRESHOLD:
            print(f"Handling intent: {predicted_tag} (Confidence: {confidence:.2f})")

            # --- Specific Intent Actions ---
            if predicted_tag == 'goodbye':
                response = self.get_response(predicted_tag)
                speak(response)
                self.gui.root.quit()
                sys.exit(0)

            elif predicted_tag == 'web_search':
                 query = self.extract_search_query(lowered_cmd)
                 if not query: # If extraction failed or resulted in empty query
                     speak("What exactly would you like me to search for?")
                     query_clarification = listen()
                     if query_clarification:
                         query = query_clarification.strip().lower()
                     else:
                         speak("Okay, cancelling the search.")
                         return # Exit handling if no clarification
                 if query:
                    response_kwargs['search_query'] = query
                    response = self.get_response(predicted_tag, **response_kwargs)
                    speak(response)
                    try:
                        webbrowser.open(f"https://google.com/search?q={query.replace(' ','+')}") # URL encode query
                    except Exception as e:
                        print(f"Error opening browser: {e}")
                        speak("Sorry, I encountered an error trying to open the web browser.")
                 else:
                     # Should not happen due to clarification step, but as fallback
                     speak("It seems I couldn't determine what to search for.")
                 return

            elif predicted_tag == 'weather':
                location = self.extract_location(lowered_cmd) # Basic location extraction
                weather_data = self.get_weather(location) # Call implemented function

                if weather_data:
                    response_kwargs.update(weather_data) # Add temp, conditions etc.
                    # weather_info can contain a summary sentence
                    response = self.get_response(predicted_tag, **response_kwargs)
                else:
                    # API failed, get generic weather response or failure message
                    response = f"Sorry, I couldn't retrieve the weather information for {location or 'your area'} right now."
                speak(response)
                return

            elif predicted_tag == 'time':
                response = self.get_response(predicted_tag) # Gets formatted time
                speak(response)
                return

            elif predicted_tag == 'date':
                 response = self.get_response(predicted_tag) # Gets formatted date
                 speak(response)
                 return

            elif predicted_tag == 'open_file':
                 filename = self.extract_filename(lowered_cmd)
                 if not filename:
                     speak("Please specify which file you want to open.")
                     # Potentially listen again for filename clarification
                     return
                 response_kwargs['filename'] = filename
                 response = self.get_response(predicted_tag, **response_kwargs)
                 speak(response)
                 self.open_file_action(filename) # Separate action function
                 return

            # Add other specific intents here (e.g., music control)

            # --- General Intent Response ---
            else:
                # For intents that just need a canned response (greeting, thanks, etc.)
                response = self.get_response(predicted_tag)
                speak(response)
                return

        # --- Handling "Unknown" Intent or Low Confidence Fallback ---
        elif predicted_tag == "unknown" and confidence >= CONFIDENCE_THRESHOLD:
            print(f"Handling 'unknown' intent confidently (Confidence: {confidence:.2f})")
            response = self.get_response("unknown") # Get specific unknown response
            speak(response)
            return

        else: # Low confidence for any tag
            print(f"Intent '{predicted_tag}' below threshold ({confidence:.2f} < {CONFIDENCE_THRESHOLD}) or no prediction. Trying keyword fallbacks.")

            # --- Keyword Fallbacks (Keep essential ones) ---
            # (Example: Keeping 'open' keyword as potentially more reliable than ML for now)
            if 'open ' in lowered_cmd: # Check if 'open ' is present, not just startswith
                 filename = lowered_cmd.split('open ', 1)[-1].strip()
                 if filename:
                     speak(f"Okay, trying the keyword 'open' for {filename}.")
                     self.open_file_action(filename)
                     return
                 else:
                      speak("You said 'open', but didn't specify a file.")
                      return

            # Maybe add very specific, high-confidence keywords if needed

            # --- Final Fallback (If Intent unclear AND no keyword match)---
            speak(random.choice([
                "Sorry, I'm not quite sure what you mean by that.",
                "I didn't understand that. Could you please rephrase?",
                "I'm still under development for that kind of request.",
                "Hmm, that's a bit unclear to me right now."
            ]))


    # --- Action & Extraction Helper Functions ---

    def open_file_action(self, filename):
        """Finds and opens the specified file."""
        paths = self.fm.find_paths(filename)
        if paths:
            filepath = paths[0]
            speak(f"Found it. Opening {os.path.basename(filepath)}")
            try:
                # Use os.startfile on Windows, 'open' on macOS, 'xdg-open' on Linux
                if sys.platform == "win32":
                    os.startfile(filepath)
                elif sys.platform == "darwin":
                    subprocess.call(["open", filepath])
                else:
                    subprocess.call(["xdg-open", filepath])
            except FileNotFoundError:
                 speak(f"I found the file path, but couldn't find an application to open '{filename}'.")
            except Exception as e:
                print(f"Error opening file '{filepath}': {e}")
                speak(f"Sorry, I encountered an error trying to open {filename}.")
        else:
            speak(f"Sorry, I couldn't find a file named '{filename}' in the indexed locations.")


    def extract_search_query(self, text):
        """More robust extraction for search queries."""
        # List of trigger phrases (more comprehensive)
        triggers = [
            r"search for", r"look up", r"find information about", r"google",
            r"search the web for", r"search online for", r"find me information on",
            r"find", r"web search", r"browse for", r"research", r"can you google",
            r"can you look up", r"i need details about", r"what is", r"who is",
            r"tell me about", r"give me more information about", r"show me"
        ]
        # Sort triggers by length descending to match longer phrases first
        triggers.sort(key=len, reverse=True)

        query = text.lower()
        for trigger in triggers:
             # Use regex to match whole words/phrases at the beginning
             match = re.match(r"\b" + trigger + r"\b\s*(.*)", query)
             if match:
                 extracted_query = match.group(1).strip()
                 # Avoid returning empty strings if trigger was the whole command
                 if extracted_query:
                      print(f"Extracted query '{extracted_query}' using trigger '{trigger}'")
                      return extracted_query
        # If no trigger phrase matched at the start, assume the whole text is the query,
        # but only if it doesn't seem like another command (simple check)
        if not any(cmd in query for cmd in ["open", "what time", "what date", "weather"]):
             print(f"No trigger phrase found, assuming full command is query: '{query}'")
             return query
        return None # Indicate query couldn't be reliably extracted


    def extract_filename(self, text):
        """Basic filename extraction (assumes format like 'open [filename]')."""
        # Use regex looking for keywords followed by potential filename chars
        # This is still basic and might capture too much/little
        match = re.search(r"\b(?:open|launch|start)\b\s+(.+)", text.lower())
        if match:
            filename = match.group(1).strip()
             # Simple cleanup (remove potential trailing punctuation if needed)
            filename = filename.rstrip('?.!')
            print(f"Extracted filename: {filename}")
            return filename
        return None


    def extract_location(self, text):
        """Placeholder for location extraction (basic)."""
        # Very simple: look for "in [location]" or "[location] weather"
        match_in = re.search(r"\bweather\b.*\bin\b\s+([a-zA-Z\s]+)", text.lower())
        if match_in:
            location = match_in.group(1).strip()
            print(f"Extracted location (using 'in'): {location}")
            return location

        match_loc_first = re.search(r"([a-zA-Z\s]+)\s+\bweather\b", text.lower())
        if match_loc_first:
             # Avoid capturing generic phrases like "what's the"
             potential_loc = match_loc_first.group(1).strip()
             if potential_loc not in ["what's the", "tell me the", "current"]:
                  print(f"Extracted location (using 'location weather'): {potential_loc}")
                  return potential_loc

        print("Could not extract specific location, will use default.")
        return None # Return None to use default/ask user

    def get_weather(self, location=None):
        """Fetches weather from OpenWeatherMap API."""
        if WEATHER_API_KEY == "YOUR_OPENWEATHERMAP_API_KEY" or not WEATHER_API_KEY:
            print("Warning: OpenWeatherMap API key not set.")
            return None

        if not location:
             # Attempt to get location based on IP (requires additional libraries like 'requests' and 'geocoder')
             # For simplicity, using a default city or asking user is often better.
             # Defaulting to a major Indian city for now as per context.
             location = "Delhi" # Default location
             print(f"No location specified, defaulting to {location}.")


        complete_url = f"{WEATHER_BASE_URL}appid={WEATHER_API_KEY}&q={location}&units=metric" # Use metric units

        try:
            response = requests.get(complete_url, timeout=10) # Add timeout
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            data = response.json()

            if data.get("cod") != 200: # Check API specific code
                print(f"Weather API Error for {location}: {data.get('message', 'Unknown error')}")
                return None

            main = data.get("main")
            weather = data.get("weather")[0] if data.get("weather") else {}
            wind = data.get("wind")

            if not main or not weather:
                print(f"Weather data incomplete for {location}.")
                return None

            temperature = main.get("temp")
            feels_like = main.get("feels_like")
            humidity = main.get("humidity")
            description = weather.get("description")
            wind_speed = wind.get("speed") # meters per second

            # Prepare info string and dictionary
            info = (
                f"Currently in {location.title()}, it feels like {feels_like:.1f}°C "
                f"with {description}. Humidity is at {humidity}%."
                # f" Wind speed is {wind_speed * 3.6:.1f} km/h." # Convert m/s to km/h
            )

            weather_dict = {
                "temperature": f"{temperature:.1f}°C",
                "conditions": description,
                "weather_info": info,
                # Add more fields if needed by response placeholders
            }
            return weather_dict

        except requests.exceptions.RequestException as e:
             print(f"Network error fetching weather for {location}: {e}")
             return None
        except json.JSONDecodeError:
             print(f"Error decoding weather API response for {location}.")
             return None
        except Exception as e:
             print(f"Unexpected error fetching weather: {e}")
             return None


# --- Main Execution ---
if __name__ == "__main__":
    # Use importlib to find SelfAttention if needed - avoids circular dependency
    from importlib import import_module
    try:
        train_module = import_module("train_model")
        SelfAttention = getattr(train_module, "SelfAttention", None)
        if SelfAttention:
             tf.keras.utils.get_custom_objects()['SelfAttention'] = SelfAttention
    except ImportError:
         print("Could not import SelfAttention from train_model. Assuming it's not needed or built-in.")
    except AttributeError:
           print("SelfAttention class not found in train_model. Assuming it's not needed or built-in.")

    try:
        root = tk.Tk()
        gui = AssistantGUI(root)
        assistant = Assistant(gui)
        assistant_thread = threading.Thread(target=assistant.run, daemon=True)
        assistant_thread.start()
        root.mainloop()

    except Exception as e:
        print(f"An error occurred during startup: {e}")
        if 'engine' in globals() and engine:
            speak("A critical error occurred during startup.")