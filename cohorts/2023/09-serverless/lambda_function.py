import numpy as np
import urllib.request
from PIL import Image
import tflite_runtime.interpreter as tflite

def download_image(image_url):
    """Downloads an image from a URL and returns it as a PIL Image."""
    with urllib.request.urlopen(image_url) as response:
        return Image.open(response)

def preprocess_image(image, target_size=(150, 150)):
    """Preprocess the image to fit the model's input requirements."""
    image = image.resize(target_size)
    image = np.array(image, dtype=np.float32)
    image = preprocess_input(image)  # Adjust this based on your model's needs
    return image

def preprocess_input(x):
    """Preprocess input (example: normalization)."""
    x /= 127.5
    x -= 1.
    return x

def load_model(model_path='bees-wasps-v2.tflite'):
    """Load the TFLite model."""
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def predict(interpreter, input_data):
    """Run prediction on the input data using the interpreter."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])

# Main execution flow
if __name__ == "__main__":
    image_url = "https://habrastorage.org/webt/rt/d9/dh/rtd9dhsmhwrdezeldzoqgijdg8a.jpeg"
    image = download_image(image_url)
    preprocessed_image = preprocess_image(image)

    input_data = np.expand_dims(preprocessed_image, axis=0)
    interpreter = load_model()
    prediction = predict(interpreter, input_data)

    print(f"Model prediction: {prediction}")
