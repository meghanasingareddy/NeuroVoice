from flask import Flask, request, jsonify, render_template, send_from_directory
import tensorflow as tf
from werkzeug.utils import secure_filename
from gtts import gTTS
import os
import uuid
import numpy as np

from model_loader import load_model_and_tokenizer
from preprocessing import preprocess_eeg_for_prediction
from preprocessing import MAX_SENTENCE_LENGTH

app = Flask(__name__)

# --- Load Model and Create Inference Models ---
training_model, tokenizer = load_model_and_tokenizer()

if training_model and tokenizer:
    # Reverse-lookup token index to decode sequences back to something readable.
    reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}
    
    # --- Define Inference Encoder Model ---
    # The encoder's input is the first input of the training model (EEG data)
    encoder_inputs = training_model.input[0]
    # The encoder's states are the outputs of the encoder LSTM layer (index 3)
    _, state_h_enc, state_c_enc = training_model.layers[3].output
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = tf.keras.Model(encoder_inputs, encoder_states)

    # --- Define Inference Decoder Model ---
    # Create new inputs for the inference decoder
    decoder_inputs_inf = tf.keras.layers.Input(shape=(1,), name='decoder_input_inf')
    decoder_state_input_h = tf.keras.layers.Input(shape=(256,), name='decoder_state_h')
    decoder_state_input_c = tf.keras.layers.Input(shape=(256,), name='decoder_state_c')
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    # Get layers from the training model
    decoder_embedding_layer = training_model.layers[2]
    decoder_lstm_layer = training_model.layers[4]
    decoder_dense_layer = training_model.layers[5]
    
    # Reconstruct the decoder for inference
    decoder_embedded = decoder_embedding_layer(decoder_inputs_inf)
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm_layer(
        decoder_embedded, initial_state=decoder_states_inputs
    )
    decoder_states = [state_h_dec, state_c_dec]
    decoder_outputs = decoder_dense_layer(decoder_outputs)
    
    decoder_model = tf.keras.Model(
        [decoder_inputs_inf] + decoder_states_inputs, [decoder_outputs] + decoder_states
    )
else:
    encoder_model, decoder_model, reverse_word_index = None, None, None

# Create a directory for generated audio files if it doesn't exist
if not os.path.exists('static/audio'):
    os.makedirs('static/audio')

def decode_sequence(input_seq):
    """Decodes an input EEG sequence into a sentence using the inference models."""
    if not all([training_model, tokenizer, encoder_model, decoder_model, reverse_word_index]):
        return "Model or tokenizer not loaded properly."

    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))
    # Populate the first token of target sequence with the start token.
    target_seq[0, 0] = tokenizer.word_index['<start>']

    stop_condition = False
    decoded_sentence = ''
    
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = reverse_word_index.get(sampled_token_index, '<unk>')
        
        if sampled_word == '<end>' or len(decoded_sentence.split()) > MAX_SENTENCE_LENGTH:
            stop_condition = True
        else:
            decoded_sentence += ' ' + sampled_word

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence.strip()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'eeg_file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['eeg_file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and file.filename.endswith('.cnt'):
        filepath = None
        try:
            upload_folder = 'uploads'
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)
            
            filename = secure_filename(file.filename)
            filepath = os.path.join(upload_folder, filename)
            file.save(filepath)

            processed_data = preprocess_eeg_for_prediction(filepath)
            
            if processed_data is None:
                return jsonify({'error': 'Failed to preprocess EEG data'}), 500

            # --- Make a prediction using the decoding function ---
            predicted_sentence = decode_sequence(processed_data)

            # Convert text to speech
            tts = gTTS(text=predicted_sentence, lang='en')
            
            audio_filename = f"{uuid.uuid4()}.mp3"
            audio_path = os.path.join('static', 'audio', audio_filename)
            tts.save(audio_path)
            
            return jsonify({
                'sentence': predicted_sentence,
                'audio_url': f"/voice/{audio_filename}"
            })

        except Exception as e:
            return jsonify({'error': f'An error occurred: {str(e)}'}), 500
        finally:
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
            
    return jsonify({'error': 'Invalid file type. Please upload a .cnt file'}), 400

@app.route('/voice/<filename>')
def get_voice(filename):
    return send_from_directory(os.path.join('static', 'audio'), filename)

if __name__ == '__main__':
    app.run()