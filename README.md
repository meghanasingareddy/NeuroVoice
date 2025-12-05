# NeuroVoice

This project is a web application that predicts text from EEG data and converts it to speech.

## Project Structure

```
.
├── app.py              # Main Flask application
├── create_model.py     # Script to create a dummy model
├── model_loader.py     # Utility to load the trained model
├── preprocessing.py    # EEG data preprocessing functions
├── README.md           # This file
├── requirements.txt    # Python dependencies
├── static/
│   ├── script.js       # Frontend JavaScript
│   └── styles.css      # Frontend CSS
└── templates/
    └── index.html      # Main HTML page
```

## Setup and Running the Application

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Create the Model:**
    Run the `create_model.py` script to generate a dummy `model.h5` file.
    ```bash
    python create_model.py
    ```

3.  **Run the Application:**
    ```bash
    python app.py
    ```

4.  **Access the Application:**
    Open your web browser and go to `http://127.0.0.1:5000`.