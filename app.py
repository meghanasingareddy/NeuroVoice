from flask import Flask, request, jsonify, render_template, send_from_directory
from gtts import gTTS
import os
import uuid

from model_loader import load_model
from preprocessing import preprocess_eeg

app = Flask(__name__)
model = load_model()

# Create a directory for generated audio files if it doesn't exist
if not os.path.exists('static/audio'):
    os.makedirs('static/audio')

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
        try:
            # Save the uploaded file temporarily
            upload_folder = 'uploads'
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)
            
            filepath = os.path.join(upload_folder, file.filename)
            file.save(filepath)

            # Preprocess the EEG data
            processed_data = preprocess_eeg(filepath)
            
            if processed_data is None:
                return jsonify({'error': 'Failed to preprocess EEG data'}), 500

            # Make a prediction
            # This is a placeholder for the actual prediction logic
            prediction_result = model.predict(processed_data)
            predicted_sentence = "This is a predicted sentence from the EEG data."

            # Convert text to speech
            tts = gTTS(text=predicted_sentence, lang='en')
            
            # Generate a unique filename for the audio
            audio_filename = f"{uuid.uuid4()}.mp3"
            audio_path = os.path.join('static', 'audio', audio_filename)
            tts.save(audio_path)
            
            # Clean up the uploaded file
            os.remove(filepath)
            
            return jsonify({
                'sentence': predicted_sentence,
                'audio_url': f"/voice/{audio_filename}"
            })

        except Exception as e:
            return jsonify({'error': f'An error occurred: {str(e)}'}), 500
            
    return jsonify({'error': 'Invalid file type. Please upload a .cnt file'}), 400

@app.route('/voice/<filename>')
def get_voice(filename):
    return send_from_directory(os.path.join('static', 'audio'), filename)

if __name__ == '__main__':
    app.run(debug=True)