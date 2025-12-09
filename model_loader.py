import tensorflow as tf
import pickle

def load_model_and_tokenizer(model_path='model.h5', tokenizer_path='tokenizer.pkl'):
    """Loads the trained Keras model and the tokenizer."""
    model = None
    tokenizer = None
    
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        
    try:
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        print("Tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        
    return model, tokenizer

# For backwards compatibility, but recommend using the function above
def load_model(model_path='model.h5'):
    """Loads the trained Keras model."""
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None