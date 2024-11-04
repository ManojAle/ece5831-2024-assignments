import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle

class Mnist():
    """
    A class to represent an MNIST-like neural network model for digit classification.
    """

    def __init__(self):
        """
        Initializes an instance of the Mnist class, preparing a dictionary to hold the model parameters.
        """
        self.params = {}

    def sigmoid(self, x):
        """
        Applies the sigmoid activation function element-wise.

        Parameters:
        x (ndarray): Input array.

        Returns:
        ndarray: Output array after applying sigmoid function.
        """
        return 1 / (1 + np.exp(-x))

    def softmax(self, a):
        """
        Applies the softmax function to an array, normalizing it into a probability distribution.

        Parameters:
        a (ndarray): Input array.

        Returns:
        ndarray: Softmax-transformed array representing probabilities.
        """
        c = np.max(a)
        exp_a = np.exp(a - c)
        return exp_a / np.sum(exp_a)
    
    def init_network(self):
        """
        Loads the pre-trained network parameters (weights and biases) from a .pkl file.
        """
        with open('Model/sample_weight.pkl', 'rb') as f:
            self.params = pickle.load(f)

    def predict(self, x):
        """
        Performs a forward pass through the network to predict the digit.

        Parameters:
        x (ndarray): Input array of image data, flattened to (1, 784) for a 28x28 image.

        Returns:
        ndarray: Output array representing the predicted probabilities for each digit (0-9).
        """
        w1, w2, w3 = self.params['W1'], self.params['W2'], self.params['W3']
        b1, b2, b3 = self.params['b1'], self.params['b2'], self.params['b3']

        a1 = np.dot(x, w1) + b1
        z1 = self.sigmoid(a1)

        a2 = np.dot(z1, w2) + b2
        z2 = self.sigmoid(a2)

        a3 = np.dot(z2, w3) + b3
        y = self.softmax(a3)

        return y    

def main():
    """
    The main function that loads an image, preprocesses it, and uses the Mnist model to predict its digit class.
    It displays the image and prediction result, and outputs whether the prediction was correct.

    Command-line Arguments:
    image_filename (str): The path to the image file.
    actual_digit (int): The true digit label for the image.

    Usage:
    python module5-3.py <image_filename> <digit>
    """
    if len(sys.argv) < 3:
        print("Usage: python module5-3.py <image_filename> <digit>")
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

    # Resize and normalize
    image = cv2.resize(image, (28, 28))
    image = image.reshape(1, 28 * 28).astype(np.float32) / 255.0

    # Initialize Mnist class and make a prediction
    mnist_model = Mnist()
    mnist_model.init_network()
    predicted_digit = np.argmax(mnist_model.predict(image))

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
