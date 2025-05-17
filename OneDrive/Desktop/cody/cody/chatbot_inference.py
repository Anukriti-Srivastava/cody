import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer # Import Layer for custom objects

# --- Custom Layers needed for loading the model ---
# Make sure the definition of SelfAttention matches exactly the one in train_model.py
class SelfAttention(Layer):
    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        if not isinstance(self.units, int) or self.units <= 0:
            raise ValueError("Attention 'units' must be a positive integer.")

    def build(self, input_shape):
        if len(input_shape) != 3:
             raise ValueError("Input to SelfAttention must be 3D (batch_size, sequence_length, features).")
        last_dim = input_shape[-1]
        if last_dim is None:
             raise ValueError("The last dimension of the input to SelfAttention cannot be None.")

        self.W1 = self.add_weight(
            shape=(last_dim, self.units),
            initializer='glorot_uniform',
            trainable=True,
            name="Att_W1"
        )
        self.W2 = self.add_weight(
            shape=(self.units,),
            initializer='glorot_uniform',
            trainable=True,
            name="Att_W2"
        )
        super().build(input_shape)

    def call(self, inputs):
        score_first_part = tf.nn.tanh(tf.tensordot(inputs, self.W1, axes=1))
        attention_scores = tf.tensordot(score_first_part, tf.expand_dims(self.W2, axis=-1), axes=1)[:,:,0]
        attention_weights = tf.nn.softmax(attention_scores, axis=1)
        weighted_input = inputs * tf.expand_dims(attention_weights, axis=-1)
        weighted_sum = tf.reduce_sum(weighted_input, axis=1)
        return weighted_sum

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config
# --- End Custom Layers ---


# --- Configuration ---
MODEL_DIR = 'saved_models'
MODEL_PATH = f'{MODEL_DIR}/best_intent_model.h5'
TOKENIZER_PATH = f'{MODEL_DIR}/tokenizer.pkl'
LABEL_ENCODER_PATH = f'{MODEL_DIR}/label_encoder.pkl'
TRAINING_CONFIG_PATH = f'{MODEL_DIR}/training_config.json' # To get max_len etc.
FEEDBACK_MEMORY_PATH = 'feedback_memory.json' # File to store feedback

# --- Global Variables (Loaded Artifacts and Memory) ---
model = None
tokenizer = None
label_encoder = None
max_len = None
feedback_memory = {} # Dictionary to store {cleaned_query: good_response}

# --- Helper Functions ---
def load_artifacts():
    """Loads the trained model, tokenizer, label encoder, and config."""
    global model, tokenizer, label_encoder, max_len

    print(f"Loading model from {MODEL_PATH}...")
    try:
        # Need custom_objects dictionary to load the model with custom layers
        model = load_model(MODEL_PATH, custom_objects={'SelfAttention': SelfAttention})
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}. Please ensure training was successful.")
        return False
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

    print(f"Loading tokenizer from {TOKENIZER_PATH}...")
    try:
        with open(TOKENIZER_PATH, 'rb') as f:
            tokenizer = pickle.load(f)
        print("Tokenizer loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Tokenizer file not found at {TOKENIZER_PATH}. Please ensure training was successful.")
        return False
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return False

    print(f"Loading label encoder from {LABEL_ENCODER_PATH}...")
    try:
        with open(LABEL_ENCODER_PATH, 'rb') as f:
            label_encoder = pickle.load(f)
        print("Label encoder loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Label encoder file not found at {LABEL_ENCODER_PATH}. Please ensure training was successful.")
        return False
    except Exception as e:
        print(f"Error loading label encoder: {e}")
        return False

    print(f"Loading training config from {TRAINING_CONFIG_PATH}...")
    try:
        with open(TRAINING_CONFIG_PATH, 'r') as f:
            training_config = json.load(f)
            max_len = training_config.get('max_len', 25) # Default to 25 if not in config
        print(f"Training config loaded. Max length set to {max_len}.")
    except FileNotFoundError:
        print(f"Warning: Training config not found at {TRAINING_CONFIG_PATH}. Using default max_len={max_len}.")
    except Exception as e:
        print(f"Error loading training config: {e}")
        # Continue with default max_len

    return True

def load_feedback_memory():
    """Loads feedback memory from a JSON file."""
    global feedback_memory
    try:
        with open(FEEDBACK_MEMORY_PATH, 'r') as f:
            feedback_memory = json.load(f)
        print(f"Feedback memory loaded from {FEEDBACK_MEMORY_PATH} ({len(feedback_memory)} entries).")
    except FileNotFoundError:
        print(f"Feedback memory file not found at {FEEDBACK_MEMORY_PATH}. Starting with empty memory.")
        feedback_memory = {}
    except json.JSONDecodeError:
        print(f"Error decoding feedback memory from {FEEDBACK_MEMORY_PATH}. Starting with empty memory.")
        feedback_memory = {}
    except Exception as e:
        print(f"Error loading feedback memory: {e}. Starting with empty memory.")
        feedback_memory = {}


def save_feedback_memory():
    """Saves current feedback memory to a JSON file."""
    try:
        with open(FEEDBACK_MEMORY_PATH, 'w') as f:
            json.dump(feedback_memory, f, indent=4)
        print(f"Feedback memory saved to {FEEDBACK_MEMORY_PATH}.")
    except Exception as e:
        print(f"Error saving feedback memory: {e}")

def clean_query(query):
    """Basic cleaning for query comparison/storage."""
    # Add any necessary cleaning steps here (lowercase, remove punctuation etc.)
    return query.lower().strip()

def predict_intent(query):
    """Uses the trained model to predict intent."""
    if tokenizer is None or model is None or label_encoder is None or max_len is None:
        print("Error: Model artifacts not loaded.")
        return "Error", "Could not process query."

    # Preprocess the query
    cleaned_query = clean_query(query)
    seq = tokenizer.texts_to_sequences([cleaned_query])
    padded_seq = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')

    # Predict
    predictions = model.predict(padded_seq, verbose=0)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions, axis=1)[0]
    predicted_tag = label_encoder.classes_[predicted_class_index]

    return predicted_tag, confidence

def get_bot_response(intent_tag):
    """
    Simulates generating a response based on the predicted intent.
    Replace this with your actual response generation logic.
    """
    # In a real bot, you would look up responses in your intents.json
    # or use a more complex generation method based on the tag.
    # For this example, we'll just return a placeholder.
    # You could load the intents.json here and pick a random response for the tag.
    try:
        with open('intents.json') as f:
            data = json.load(f)
            for intent in data['intents']:
                if intent['tag'] == intent_tag and 'responses' in intent:
                    # Pick a random response from the intents file
                    return random.choice(intent['responses'])
    except FileNotFoundError:
        print("Warning: intents.json not found. Cannot retrieve specific responses.")
        return f"Understood intent: {intent_tag}."
    except Exception as e:
        print(f"Warning: Error retrieving response from intents.json: {e}")
        return f"Understood intent: {intent_tag}."

    return f"Understood intent: {intent_tag}. (No specific response found in intents.json)"


# --- Main Chatbot Loop ---
def run_chatbot():
    """Main function to run the chatbot with feedback loop."""
    if not load_artifacts():
        print("Failed to load model artifacts. Exiting.")
        return

    load_feedback_memory()

    print("\nChatbot is ready! Type 'quit' to exit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break

        cleaned_input = clean_query(user_input)

        # --- Feedback Loop Lookup ---
        if cleaned_input in feedback_memory:
            bot_response = feedback_memory[cleaned_input]
            print(f"Bot (Cached): {bot_response}")
            # For cached responses, we assume they were 'good', no need for feedback
            continue # Skip prediction and feedback steps

        # --- Normal Prediction and Response Generation ---
        predicted_tag, confidence = predict_intent(user_input) # Use original input for clarity in get_bot_response potentially

        # Check confidence threshold if needed (optional)
       # if confidence < 0.4:
             bot_response = "I'm not sure I understand. Can you rephrase?"
             print(f"Bot: {bot_response} (Confidence: {confidence:.2f})")
        #else:
        bot_response = get_bot_response(predicted_tag) # Generate response based on predicted tag
        print(f"Bot: {bot_response} (Intent: {predicted_tag}, Confidence: {confidence:.2f})")


        # --- Ask for Feedback ---
        while True: # Loop until valid feedback is given
            feedback = input("Was this a good response? (y/n): ").lower().strip()
            if feedback in ['y', 'yes', 'n', 'no']:
                break
            else:
                print("Please enter 'y' or 'n'.")

        if feedback in ['y', 'yes']:
            # Store the query and the successful response in memory
            feedback_memory[cleaned_input] = bot_response
            print("Thanks for the feedback! I'll remember that response for this question.")
            save_feedback_memory() # Save memory after adding an entry

        # 'n' feedback is noted, but doesn't add to the positive memory cache in this basic version
    print("Exiting Chatbot. Saving final feedback memory...")
    save_feedback_memory()
    print("Memory saved. Goodbye!")

# --- Script Entry Point ---
if __name__ == '__main__':
    import random # Make random available for get_bot_response example
    run_chatbot()