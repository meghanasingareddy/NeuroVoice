import tensorflow as tf

def load_model(model_path='model.h5'):
    """Loads the trained Keras model."""
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None