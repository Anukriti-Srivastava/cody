import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dropout, Dense, GlobalAveragePooling1D, Layer, GRU
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import backend as K
import nltk
from nltk.corpus import wordnet
import random
import os
from datetime import datetime
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import re # Added for pattern cleaning

# Ensure NLTK data is downloaded
try:
    nltk.data.find('corpora/wordnet.zip')
    print("NLTK 'wordnet' data found.")
except LookupError:
    print("Downloading NLTK 'wordnet' data...")
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4.zip')
    print("NLTK 'omw-1.4' data found.")
except LookupError:
    print("Downloading NLTK 'omw-1.4' data...")
    nltk.download('omw-1.4')
# --------------------------

# -------------------- CONFIG --------------------
CONFIG = {
    'intents_file': 'intents.json',
    'glove_path': 'glove.6B.100d.txt',  # Make sure this path is correct
    'vocab_size': 5000,
    'embedding_dim': 100,
    'max_len': 25, # Ensure this matches MAX_LEN if used elsewhere
    'validation_split': 0.2,
    'learning_rate': 5e-4,
    'batch_size': 128,
    'epochs': 100,
    'model_checkpoint_path': 'model_checkpoints/best_intent_model.h5',
    'model_output_dir': 'saved_models',
    'augmentation_factor': 3 # Factor by which to multiply data via augmentation
}

# -------------------- LOGGING --------------------
def setup_logging():
    """Sets up logging to both a file and the console."""
    os.makedirs('logs', exist_ok=True)
    import logging
    log_filename = f"logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler() # Also print logs to console
        ]
    )
    print(f"Logging to {log_filename}")
    return logging.getLogger(__name__)

logger = setup_logging()

# -------------------- DATA AUGMENTATION --------------------
def synonym_augment(sentence):
    """
    Augments a sentence by replacing a random percentage of words with synonyms.
    """
    words = sentence.split()
    # Create a list of (word, index) tuples to keep track of original positions
    word_indices = list(enumerate(words))
    random.shuffle(word_indices) # Shuffle for random selection
    
    new_words = list(words) # Start with the original words list
    num_replaced = 0
    # Replace up to 20% of words, minimum 1
    max_replace = max(1, int(len(words) * 0.2)) 

    for original_index, word in word_indices:
        if num_replaced >= max_replace:
            break
            
        syns = wordnet.synsets(word)
        if syns:
            # Get all unique lemmas from all synsets for the word
            lemmas = [l.name().replace('_', ' ') for s in syns for l in s.lemmas()]
            # Exclude the original word itself
            unique_synonyms = list(set(lemmas) - set([word]))
            
            if unique_synonyms:
                synonym = random.choice(unique_synonyms)
                # Replace the word at its original index in the new_words list
                new_words[original_index] = synonym
                num_replaced += 1
                
    augmented_sentence = ' '.join(new_words)
    # Return augmented sentence only if it's different from the original
    return augmented_sentence if augmented_sentence != sentence else None


# -------------------- DATA LOADING & PREPROCESSING --------------------
def load_and_augment_data(config):
    """
    Loads intents data, performs basic cleaning, applies data augmentation,
    tokenizes, pads sequences, and encodes labels.
    """
    try:
        with open(config['intents_file']) as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Intents file not found at {config['intents_file']}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {config['intents_file']}")
        raise

    sentences, labels = [], []
    tag_set = set()
    initial_pattern_count = 0

    for intent in data.get('intents', []): # Use .get for safety
        if 'tag' not in intent or 'patterns' not in intent:
            logger.warning(f"Skipping invalid intent structure: {intent}")
            continue
        tag = intent['tag']
        tag_set.add(tag)
        for pat in intent.get('patterns', []): # Use .get for safety
            initial_pattern_count += 1
            # Basic cleaning: lowercase and strip whitespace
            cleaned_pat = str(pat).lower().strip() # Ensure it's a string

            if not cleaned_pat:
                logger.warning(f"Skipping empty pattern for tag '{tag}'")
                continue

            # Remove placeholders like {filename}, {name}, etc.
            cleaned_pat_no_placeholder = re.sub(r'\{.*?\}', '', cleaned_pat).strip()

            if not cleaned_pat_no_placeholder:
                 logger.warning(f"Skipping pattern that became empty after placeholder removal for tag '{tag}': '{pat}'")
                 continue


            sentences.append(cleaned_pat_no_placeholder)
            labels.append(tag)

            # Augmentation
            if config['augmentation_factor'] > 0:
                for _ in range(config['augmentation_factor']):
                    augmented_sentence = synonym_augment(cleaned_pat_no_placeholder)
                    # Add augmented sentence only if it's valid (not None) and different
                    if augmented_sentence is not None and augmented_sentence != cleaned_pat_no_placeholder:
                        sentences.append(augmented_sentence)
                        labels.append(tag)

    if not sentences:
        logger.error("No valid sentences found after loading and cleaning data. Check intents file structure and content.")
        raise ValueError("No training data available.")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    num_classes = len(tag_set)

    # Initialize tokenizer with OOV token
    tokenizer = Tokenizer(num_words=config['vocab_size'], oov_token='<OOV>')
    tokenizer.fit_on_texts(sentences)

    # Check if tokenizer vocabulary is empty
    if not tokenizer.word_index:
         logger.error("Tokenizer vocabulary is empty. Check if sentences contain actual words after cleaning.")
         raise ValueError("Tokenizer vocabulary is empty.")

    seqs = tokenizer.texts_to_sequences(sentences)

    # Pad sequences
    X = pad_sequences(seqs, maxlen=config['max_len'], padding='post', truncating='post')

    logger.info(f"Initial number of patterns: {initial_pattern_count}")
    logger.info(f"Loaded and augmented to {len(sentences)} sentences.")
    logger.info(f"Number of unique tags: {num_classes}")
    logger.info(f"Number of samples per tag (including augmented): {dict(zip(*np.unique(labels, return_counts=True)))}")
    logger.info(f"Vocabulary size (up to {config['vocab_size']}): {min(len(tokenizer.word_index), config['vocab_size'])}")
    logger.info(f"Label Encoder Classes: {list(label_encoder.classes_)}") # Convert to list for logging

    return X, y, tokenizer, label_encoder, num_classes

# -------------------- EMBEDDINGS --------------------
def load_glove_embeddings(config, tokenizer):
    """
    Loads GloVe embeddings and creates an embedding matrix for the tokenizer's vocabulary.
    """
    embeddings_index = {}
    try:
        with open(config['glove_path'], encoding='utf-8') as f:
            for line in f:
                values = line.split()
                if len(values) <= 1: # Skip empty or malformed lines
                    continue
                word = values[0]
                try:
                    # Ensure vector has the correct dimension
                    if len(values[1:]) == config['embedding_dim']:
                        vec = np.asarray(values[1:], dtype='float32')
                        embeddings_index[word] = vec
                    else:
                         logger.warning(f"Skipping line in GloVe file due to incorrect dimension ({len(values[1:])}): {line[:50]}...")
                         continue
                except ValueError:
                    logger.warning(f"Skipping line in GloVe file (ValueError - could not convert to float): {line[:50]}...")
                    continue # Skip lines with conversion issues
    except FileNotFoundError:
        logger.error(f"GloVe embeddings file not found at {config['glove_path']}")
        logger.warning("Proceeding without pre-trained embeddings. Model will train embeddings from scratch.")
        return None # Indicate failure to load

    logger.info(f"Found {len(embeddings_index)} word vectors in GloVe file.")

    # Initialize embedding matrix with zeros
    embedding_matrix = np.zeros((config['vocab_size'], config['embedding_dim']))
    hits = 0
    misses = 0

    # Fill the embedding matrix with GloVe vectors for words in our vocabulary
    # The first token (index 0) is typically reserved for padding, OOV is index 1 if used
    # Word index starts from 1
    for word, i in tokenizer.word_index.items():
        # Only consider words within our defined vocabulary size
        if i < config['vocab_size']:
            vec = embeddings_index.get(word)
            if vec is not None:
                # Words are 1-indexed in Keras Tokenizer by default,
                # but our embedding matrix is 0-indexed.
                # We map word index `i` to `i-1` in the embedding matrix if 0 is padding.
                # However, Keras Embedding layer handles this indexing internally
                # when initialized with `input_dim=vocab_size`.
                # So, we use the tokenizer's index `i` directly.
                embedding_matrix[i] = vec
                hits += 1
            else:
                misses += 1
    logger.info(f"Converted {hits} words ({misses} misses) from vocabulary to GloVe vectors.")

    return embedding_matrix

# -------------------- DATASET PIPELINE --------------------
def make_dataset(X, y, batch_size, shuffle=True):
    """
    Creates a TensorFlow dataset from features and labels.
    """
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        # Shuffle the dataset with a buffer size equal to the dataset size
        ds = ds.shuffle(buffer_size=len(X))
    # Batch the dataset and prefetch for performance
    return ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

# -------------------- MODEL --------------------
class SelfAttention(Layer):
    """Custom Keras Layer for Self-Attention."""
    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        # Add a check to ensure units is a positive integer
        if not isinstance(self.units, int) or self.units <= 0:
            raise ValueError("Attention 'units' must be a positive integer.")

    def build(self, input_shape):
        # input_shape is typically (batch_size, max_len, embedding_dim)
        if len(input_shape) != 3:
             raise ValueError("Input to SelfAttention must be 3D (batch_size, sequence_length, features).")

        last_dim = input_shape[-1] # Should be the feature dimension (e.g., LSTM output size)
        if last_dim is None:
             raise ValueError("The last dimension of the input to SelfAttention cannot be None.")

        # Weight matrix W1 to transform input features
        self.W1 = self.add_weight(
            shape=(last_dim, self.units),
            initializer='glorot_uniform',
            trainable=True,
            name="Att_W1"
        )
        # Context vector W2 to compute attention scores
        self.W2 = self.add_weight(
            shape=(self.units,),
            initializer='glorot_uniform',
            trainable=True,
            name="Att_W2"
        )
        super().build(input_shape)

    def call(self, inputs):
        # inputs shape: (batch_size, sequence_length, features)
        # Calculate attention scores: tanh(inputs * W1) * W2
        # inputs * W1: (batch_size, sequence_length, units)
        score_first_part = tf.nn.tanh(tf.tensordot(inputs, self.W1, axes=1))

        # score_first_part shape: (batch_size, sequence_length, units)
        # W2 shape: (units,)
        # tf.tensordot(score_first_part, W2, axes=1) performs dot product
        # over the last dimension of score_first_part and the only dimension of W2.
        # Result shape: (batch_size, sequence_length)
        # tf.expand_dims(self.W2, axis=-1) changes W2 shape to (units, 1) for the tensordot operation
        attention_scores = tf.tensordot(score_first_part, tf.expand_dims(self.W2, axis=-1), axes=1)[:,:,0]


        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(attention_scores, axis=1) # Sums to 1 across the sequence_length dimension

        # attention_weights shape: (batch_size, sequence_length)
        # inputs shape: (batch_size, sequence_length, features)

        # Apply weights to the input sequence
        # Reshape weights to (batch_size, sequence_length, 1) for element-wise multiplication
        weighted_input = inputs * tf.expand_dims(attention_weights, axis=-1)

        # Sum over the sequence length dimension to get the final context vector
        weighted_sum = tf.reduce_sum(weighted_input, axis=1)

        # weighted_sum shape: (batch_size, features) - Matches the original input features dimension
        return weighted_sum

    def compute_output_shape(self, input_shape):
        # Output shape is (batch_size, features)
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


def build_intent_model(config, num_classes, embedding_matrix=None):
    """
    Builds the Keras model for intent classification.
    """
    inp = Input(shape=(config['max_len'],), dtype='int32', name='Input')

    # Use pre-trained embeddings if available, otherwise train them
    if embedding_matrix is not None and embedding_matrix.shape == (config['vocab_size'], config['embedding_dim']):
        logger.info("Using pre-trained GloVe embeddings.")
        emb = Embedding(
            config['vocab_size'], config['embedding_dim'],
            weights=[embedding_matrix], trainable=False, name='Pretrained_Embedding')(inp)
    else:
        if embedding_matrix is not None:
             logger.warning(f"Embedding matrix shape mismatch: Expected {(config['vocab_size'], config['embedding_dim'])}, got {embedding_matrix.shape if embedding_matrix is not None else 'None'}. Training embeddings from scratch.")
        else:
             logger.warning("No pre-trained embeddings loaded. Training embeddings from scratch.")
        emb = Embedding(
            config['vocab_size'], config['embedding_dim'],
            trainable=True, name='Learned_Embedding')(inp)

    # Bidirectional GRU layer (often faster and performs similarly to LSTM)
    x = Bidirectional(GRU(128, return_sequences=True), name='BiGRU_1')(emb)
    x = Dropout(0.4, name='Dropout_1')(x) # Increased dropout slightly

    # Optional: Add a second BiGRU layer
    # x = Bidirectional(GRU(64, return_sequences=True), name='BiGRU_2')(x)
    # x = Dropout(0.3, name='Dropout_2')(x)


    # Apply Self Attention to the sequence output
    att = SelfAttention(units=64, name='Self_Attention')(x)

    # Dense layers for classification
    x = Dense(128, activation='relu', name='Dense_1')(att) # Increased dense units
    x = Dropout(0.5, name='Dropout_3')(x) # Increased final dropout

    # Output layer with softmax activation for multi-class classification
    out = Dense(num_classes, activation='softmax', name='Output')(x)

    model = Model(inputs=inp, outputs=out)

    # Define the optimizer with the specified learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])

    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy', # Use sparse_categorical_crossentropy for integer labels
        metrics=['accuracy'] # Keep metrics simple for compilation
    )
    logger.info("Model compiled successfully.")
    model.summary(print_fn=logger.info) # Log model summary
    return model

# -------------------- CALLBACKS --------------------
def create_callbacks(config):
    """
    Creates Keras callbacks for training (Early Stopping, Model Checkpointing, ReduceLROnPlateau).
    """
    checkpoint_dir = os.path.dirname(config['model_checkpoint_path'])
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger.info(f"Model checkpoints will be saved to: {checkpoint_dir}")

    callbacks = [
        EarlyStopping(
            monitor='val_loss', # Monitor validation loss
            patience=7, # Stop after 7 epochs without improvement
            restore_best_weights=True, # Restore model weights from the epoch with the best value of the monitored quantity
            verbose=1 # Print messages
        ),
        ModelCheckpoint(
            config['model_checkpoint_path'],
            monitor='val_accuracy', # Monitor validation accuracy
            save_best_only=True, # Save only when validation accuracy improves
            save_weights_only=False, # Save the entire model (architecture + weights)
            verbose=1 # Print messages
        ),
        ReduceLROnPlateau(
            monitor='val_loss', # Monitor validation loss
            factor=0.5, # Reduce learning rate by half
            patience=3, # Reduce after 3 epochs without validation loss improvement
            min_lr=1e-6, # Minimum learning rate
            verbose=1 # Print messages
        )
    ]
    return callbacks

# -------------------- SAVE ARTIFACTS --------------------
def save_artifacts(model, tokenizer, label_encoder, config):
    """
    Saves the trained model, tokenizer, label encoder, and training config.
    """
    output_dir = config['model_output_dir']
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving final model and artifacts to: {output_dir}")

    # The best model is already saved by ModelCheckpoint.
    # We can optionally save the final state of the model here, but loading
    # from the checkpoint path is recommended for prediction.
    # model.save(os.path.join(output_dir, 'final_intent_model.h5')) # Optional

    # Save the best model loaded from the checkpoint
    best_model_path_in_output_dir = os.path.join(output_dir, 'best_intent_model.h5')
    try:
        # Copy or load and save the best model to the final output directory
        # A simpler way is to just use the ModelCheckpoint path later for loading
        # but if you want a copy in the dedicated output dir:
        if os.path.exists(config['model_checkpoint_path']):
             tf.keras.models.save_model(
                 tf.keras.models.load_model(config['model_checkpoint_path'], custom_objects={'SelfAttention': SelfAttention}),
                 best_model_path_in_output_dir
                 )
             logger.info(f"Best model copied from checkpoint to {best_model_path_in_output_dir}")
        else:
             logger.warning(f"Best model checkpoint not found at {config['model_checkpoint_path']}. Skipping saving model to {output_dir}.")
    except Exception as e:
         logger.error(f"Failed to save best model to output directory: {e}")


    # Save tokenizer using pickle
    tokenizer_path = os.path.join(output_dir, 'tokenizer.pkl')
    try:
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(tokenizer, f)
        logger.info(f"Tokenizer saved to {tokenizer_path}")
    except Exception as e:
        logger.error(f"Failed to save tokenizer: {e}")


    # Save label encoder using pickle
    label_encoder_path = os.path.join(output_dir, 'label_encoder.pkl')
    try:
        with open(label_encoder_path, 'wb') as f:
            pickle.dump(label_encoder, f)
        logger.info(f"Label encoder saved to {label_encoder_path}")
    except Exception as e:
        logger.error(f"Failed to save label encoder: {e}")

    # Save config used for training
    config_path = os.path.join(output_dir, 'training_config.json')
    try:
        with open(config_path, 'w') as f:
            # Convert numpy types for JSON serialization if any present
            serializable_config = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in config.items()}
            json.dump(serializable_config, f, indent=4)
        logger.info(f"Training config saved to {config_path}")
    except Exception as e:
         logger.error(f"Failed to save training config: {e}")


# -------------------- TRAINING & EVALUATION --------------------
def train_intent_model(config):
    """
    Orchestrates the data loading, preprocessing, model building, training, and evaluation.
    """
    logger.info("--- Starting Intent Model Training ---")
    logger.info(f"Using configuration: {json.dumps(config, indent=4)}") # Log config nicely

    # Load and preprocess data
    try:
        X, y, tokenizer, label_encoder, num_classes = load_and_augment_data(config)
    except Exception as e:
        logger.error(f"Error loading and preprocessing data: {e}")
        return None # Stop execution if data loading fails

    # Check if classes are loaded correctly
    if num_classes == 0 or len(label_encoder.classes_) == 0:
        logger.error("No classes found for LabelEncoder. Check intents data.")
        return None

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=config['validation_split'], random_state=42, stratify=y) # Stratify to maintain class distribution

    logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

    # Create TensorFlow datasets
    train_ds = make_dataset(X_train, y_train, config['batch_size'], shuffle=True)
    val_ds = make_dataset(X_val, y_val, config['batch_size'], shuffle=False)

    # Load GloVe embeddings
    embedding_matrix = load_glove_embeddings(config, tokenizer) # Returns None if GloVe not found

    # Build the model
    model = build_intent_model(config, num_classes, embedding_matrix)

    # Create training callbacks
    callbacks = create_callbacks(config)

    # Start model training
    logger.info("Starting model training...")
    try:
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=config['epochs'],
            callbacks=callbacks,
            verbose=1 # Show progress bar
        )
        logger.info("Model training finished.")
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        # Attempt to save artifacts even if training failed
        save_artifacts(model, tokenizer, label_encoder, config)
        return None


    # --- Evaluation ---
    logger.info("Evaluating the best model on validation data...")
    # Load the best model saved by ModelCheckpoint
    try:
        # Need to provide custom_objects when loading a model with custom layers
        best_model = tf.keras.models.load_model(
            config['model_checkpoint_path'],
            custom_objects={'SelfAttention': SelfAttention}
        )
        logger.info(f"Successfully loaded best model from {config['model_checkpoint_path']}")

        # Evaluate loss and accuracy
        loss, accuracy = best_model.evaluate(val_ds, verbose=0)
        logger.info(f"Best Model Validation Loss: {loss:.4f}")
        logger.info(f"Best Model Validation Accuracy: {accuracy:.4f}")

        # Generate predictions for classification report
        # Predict needs the raw data, not the dataset, for sklearn report
        y_pred_probs = best_model.predict(X_val)
        y_pred = np.argmax(y_pred_probs, axis=1)

        # Generate classification report
        target_names = label_encoder.classes_
        report = classification_report(y_val, y_pred, target_names=target_names, zero_division=0)
        logger.info("Classification Report (Validation Set):\n" + report)

        # --- Save Artifacts ---
        # Save the best performing model and other artifacts
        # We pass the loaded best_model here to ensure the saved model is the best one
        save_artifacts(best_model, tokenizer, label_encoder, config)

    except FileNotFoundError:
        logger.error(f"Best model checkpoint not found at {config['model_checkpoint_path']}. Cannot perform evaluation on best model.")
        # Attempt to save artifacts from the final epoch model state if checkpoint isn't there
        logger.warning("Attempting to save artifacts from the final epoch model state (might not be the best).")
        save_artifacts(model, tokenizer, label_encoder, config) # Save final model and artifacts
    except Exception as e:
        logger.error(f"Error during model loading, evaluation or saving: {e}")
        # Attempt to save artifacts from the final epoch model state as a fallback
        logger.warning("Attempting to save artifacts from the final epoch model state as a fallback.")
        save_artifacts(model, tokenizer, label_encoder, config) # Save final model and artifacts


    logger.info("--- Intent Model Training Completed ---")
    return history # Return history for potential plotting/analysis

# -------------------- MAIN EXECUTION --------------------
if __name__ == '__main__':
    # Ensure NLTK data is available before starting training
    # (Downloads moved to top level for clarity)

    # Run the training process
    training_history = train_intent_model(CONFIG)

    if training_history:
        logger.info("Training completed successfully.")
        # You can add plotting logic here using history if needed
        # E.g., plot accuracy and loss curves
    else:
        logger.error("Training failed.")