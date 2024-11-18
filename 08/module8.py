import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from le_net import LeNet  # Import the LeNet class

def main():
    """
    The main function loads an image, preprocesses it, and uses the LeNet model to predict its digit class.
    It displays the image and prediction result, and outputs whether the prediction was correct.

    Command-line Arguments:
    image_filename (str): The path to the image file.
    actual_digit (int): The true digit label for the image.

    Usage:
    python test.py <image_filename> <digit>
    """
    if len(sys.argv) < 3:
        print("Usage: python test.py <image_filename> <digit>")
        return
    
    image_filename = sys.argv[1]
    actual_digit = int(sys.argv[2])

    # Load and preprocess the image
    image = cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Image file '{image_filename}' not found.")
        return

    # Invert colors and sharpen the image
    image = cv2.bitwise_not(image)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image = cv2.filter2D(image, -1, kernel)

    # Resize, reshape, and normalize
    image = cv2.resize(image, (28, 28))
    image = image.reshape(1, 28, 28, 1).astype(np.float32) / 255.0  # Add batch and channel dimensions

    # Load the trained LeNet model
    lenet_model = LeNet()
    lenet_model.load("Model/manoj.h5")  # Replace with your saved model name

    # Predict the digit
    predicted_digit = lenet_model.predict(image)[0]  # [0] to get the first result from batch

    # Show the image
    plt.imshow(image.reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {predicted_digit}")
    plt.show()

    # Display result
    if predicted_digit == actual_digit:
        print(f"Success: Image {image_filename} for digit {actual_digit} is recognized as {predicted_digit}.")
    else:
        print(f"Fail: Image {image_filename} is for digit {actual_digit} but the inference result is {predicted_digit}.")

if __name__ == "__main__":
    main()
