import numpy as np
import pandas as pd
import mne
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# --- Parameters ---
# TODO: Adjust these parameters based on your data and model
MAX_EEG_SEQUENCE_LENGTH = 1000 # Example: 1000 time steps
MAX_SENTENCE_LENGTH = 20 # Example: 20 words
VOCAB_SIZE = 10000 # Example: 10000 unique words

def load_eeg_data(file_path):
    """Loads and preprocesses a single .cnt file."""
    try:
        raw = mne.io.read_raw_cnt(file_path, preload=True, verbose=False)
        raw.pick_types(eeg=True)
        # TODO: Add your filtering or other preprocessing steps here
        # raw.filter(1., 40.)
        data = raw.get_data().T # Transpose to have (time_steps, channels)
        return data
    except Exception as e:
        print(f"Error loading or processing file {file_path}: {e}")
        return None

def create_tokenizer(sentences):
    """Creates and fits a tokenizer on the given sentences."""
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<unk>")
    tokenizer.fit_on_texts(sentences)
    return tokenizer

def load_and_preprocess_data_for_training(labels_csv_path, data_folder):
    """
    Loads and preprocesses the entire dataset for training a seq2seq model.
    """
    if not os.path.exists(labels_csv_path):
        print(f"Error: Labels file not found at {labels_csv_path}")
        return None, None, None, None

    labels_df = pd.read_csv(labels_csv_path)
    
    # --- 1. Load Sentences and Create Tokenizer ---
    sentences = labels_df['sentence'].tolist()
    
    # Add start and end tokens to sentences and then create the tokenizer
    sentences_with_tokens = [f"<start> {s} <end>" for s in sentences]
    tokenizer = create_tokenizer(sentences_with_tokens)
    
    tokenized_sentences = tokenizer.texts_to_sequences(sentences_with_tokens)
    
    # --- 2. Pad Sentences ---
    # The decoder input should be shifted by one time step
    decoder_input_data = pad_sequences([s[:-1] for s in tokenized_sentences], maxlen=MAX_SENTENCE_LENGTH, padding='post')
    decoder_target_data = pad_sequences([s[1:] for s in tokenized_sentences], maxlen=MAX_SENTENCE_LENGTH, padding='post')

    # --- 3. Load and Pad EEG Data ---
    eeg_data_list = []
    valid_indices = []
    for i, row in labels_df.iterrows():
        file_path = os.path.join(data_folder, row["filename"])
        eeg_data = load_eeg_data(file_path)
        if eeg_data is not None:
            eeg_data_list.append(eeg_data)
            valid_indices.append(i)
            
    if not eeg_data_list:
        print("No valid EEG data found.")
        return None, None, None, None

    encoder_input_data = pad_sequences(eeg_data_list, maxlen=MAX_EEG_SEQUENCE_LENGTH, padding='post', dtype='float32')

    # Filter the decoder data to match the valid EEG files
    decoder_input_data = decoder_input_data[valid_indices]
    decoder_target_data = decoder_target_data[valid_indices]

    print("Data shapes:")
    print("Encoder input (EEG):", encoder_input_data.shape)
    print("Decoder input (Text):", decoder_input_data.shape)
    print("Decoder target (Text):", decoder_target_data.shape)

    return encoder_input_data, decoder_input_data, decoder_target_data, tokenizer

def preprocess_eeg_for_prediction(file_path):
    """Preprocesses a single .cnt file for prediction."""
    eeg_data = load_eeg_data(file_path)
    if eeg_data is None:
        return None
        
    padded_eeg = pad_sequences([eeg_data], maxlen=MAX_EEG_SEQUENCE_LENGTH, padding='post', dtype='float32')
    return padded_eeg

if __name__ == '__main__':
    # --- Example Usage ---
    # This block demonstrates how to use the functions in this script.
    # You will need to have a 'data' folder with your .cnt files and a 'labels.csv'.
    
    DATA_FOLDER = "data"
    LABELS_CSV = os.path.join(DATA_FOLDER, "labels.csv")
    
    if os.path.exists(LABELS_CSV) and os.path.exists(DATA_FOLDER):
        print("--- Loading data for training ---")
        X_encoder, X_decoder, y_decoder, tokenizer = load_and_preprocess_data_for_training(LABELS_CSV, DATA_FOLDER)
        
        if X_encoder is not None:
            print("\n--- Preprocessing for prediction (example) ---")
            # Get the first filename from the CSV for the example
            example_filename = pd.read_csv(LABELS_CSV).iloc[0]['filename']
            example_filepath = os.path.join(DATA_FOLDER, example_filename)
            
            if os.path.exists(example_filepath):
                 processed_eeg = preprocess_eeg_for_prediction(example_filepath)
                 print(f"Shape of preprocessed EEG for prediction: {processed_eeg.shape}")
            else:
                print(f"Could not find example file for prediction: {example_filepath}")

    else:
        print("\nSkipping example usage.")
        print("Please create a 'data' folder with your .cnt files and a 'labels.csv' to run this example.")

