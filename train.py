import os
import numpy as np
import tensorflow as tf
from preprocessing import load_and_preprocess_data_for_training
from create_model import create_model
import pickle

# --- Parameters ---
DATA_FOLDER = "data"
LABELS_CSV = os.path.join(DATA_FOLDER, "labels.csv")
MODEL_SAVE_PATH = "model.h5"
TOKENIZER_SAVE_PATH = "tokenizer.pkl"
EPOCHS = 20 # TODO: Adjust the number of epochs
BATCH_SIZE = 16 # TODO: Adjust the batch size

# --- 1. Load and Preprocess Data ---
print("--- Loading and preprocessing data ---")
encoder_input_data, decoder_input_data, decoder_target_data, tokenizer = \
    load_and_preprocess_data_for_training(LABELS_CSV, DATA_FOLDER)

if encoder_input_data is None:
    print("\nError: Data loading failed. Please check your 'data' folder and 'labels.csv'.")
    print("Exiting training script.")
    exit()

# --- 2. Create the Model ---
print("\n--- Creating the model ---")
# Get parameters for the model from the preprocessed data
input_shape = encoder_input_data.shape[1:]
vocab_size = len(tokenizer.word_index) + 1
max_output_length = decoder_input_data.shape[1]

model = create_model(input_shape, vocab_size, max_output_length)
model.summary()

# --- 3. Train the Model ---
print("\n--- Starting model training ---")
# Note: The target data for a seq2seq model needs to be 3D
decoder_target_data_3d = np.expand_dims(decoder_target_data, -1)

history = model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data_3d,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.2 # Using 20% of the data for validation
)

# --- 4. Save the Model and Tokenizer ---
print("\n--- Saving model and tokenizer ---")
model.save(MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")

with open(TOKENIZER_SAVE_PATH, 'wb') as f:
    pickle.dump(tokenizer, f)
print(f"Tokenizer saved to {TOKENIZER_SAVE_PATH}")

print("\n--- Training complete ---")