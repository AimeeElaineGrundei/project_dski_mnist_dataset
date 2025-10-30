### Imports 
import keras

model_path = './models/mnist_model1.keras'

def load_model(model_path: str) -> keras.Model:
    """Load a Keras model from the specified path."""
    return keras.models.load_model(model_path)
