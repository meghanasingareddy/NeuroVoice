import tensorflow as tf
import numpy as np

def create_dummy_model():
    """Creates and saves a simple dummy model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(10,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.save('model.h5')
    print("Dummy model 'model.h5' created successfully.")

if __name__ == '__main__':
    create_dummy_model()