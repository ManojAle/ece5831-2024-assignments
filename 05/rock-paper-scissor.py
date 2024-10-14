import argparse
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

def load_image(file_path):
    """
    Load and preprocess an image for prediction.

    Parameters:
    file_path (str): The path to the image file to be loaded.

    Returns:
    Image: The loaded and converted image as an RGB image.
    """
    image = Image.open(file_path).convert("RGB")
    return image

def init():
    """
    Initialize settings for the script.
    
    Disables scientific notation for better clarity when printing numpy arrays.
    """
    np.set_printoptions(suppress=True)

def load_my_model():
    """
    Load the pre-trained model and class labels.

    Returns:
    model (keras.Model): The loaded Keras model.
    class_names (list): A list of class names corresponding to the model's predictions.
    """
    # Load the model
    model = load_model("model/keras_model.h5")

    # Load the labels
    class_names = open("model/labels.txt", "r").readlines()

    return model, class_names

def prep_input(image):
    """
    Preprocess the input image to make it suitable for model prediction.

    Parameters:
    image (PIL.Image.Image): The input image to be preprocessed.

    Returns:
    np.ndarray: A numpy array that is normalized and ready for prediction.
    """
    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Resize the image to 224x224 and crop from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # Convert the image to a numpy array
    image_array = np.asarray(image)

    # Normalize the image array
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the data array
    data[0] = normalized_image_array

    return data

def predict(model, class_names, data):
    """
    Predict the class of the input image using the pre-trained model.

    Parameters:
    model (keras.Model): The pre-trained Keras model.
    class_names (list): A list of class names corresponding to the model's predictions.
    data (np.ndarray): The preprocessed image data.

    Prints:
    The predicted class and confidence score.
    """
    # Predict the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()  # Strip to remove newline characters
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print(f"Class: {class_name}, Confidence Score: {confidence_score:.4f}")

def main():
    """
    Main entry point for the script.

    This script takes an image file path as a command line argument, loads the
    pre-trained model and class labels, processes the input image, and predicts
    the class (rock, paper, or scissors) with the associated confidence score.

    Usage:
    python rock-paper-scissor.py <path_to_image>
    """
    # Initialize argument parser
    parser = argparse.ArgumentParser(description='Classify an image as Rock, Paper, or Scissors using a pre-trained model.')

    # Adding argument for the image path
    parser.add_argument('image_path', metavar='image_path', type=str, help='The path to the image to classify.')

    # Parse arguments
    args = parser.parse_args()

    # Initialize, load the model and class names, preprocess the image, and predict
    init()
    image = load_image(args.image_path)
    model, class_names = load_my_model()
    data = prep_input(image)
    predict(model, class_names, data)

if __name__ == "__main__":
    main()
