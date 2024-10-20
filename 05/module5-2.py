import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mnist_data import MnistData  # Assuming your MnistData class is in mnist_data.py

def parse_arguments():
    """
    Parse command line arguments.
    
    The script expects two arguments:
    1. 'train' or 'test' - to specify which dataset to use.
    2. Index number to fetch an image from the dataset.
    """
    parser = argparse.ArgumentParser(description="Test MnistData class with given arguments.")
    parser.add_argument('dataset', choices=['train', 'test'], help="Choose dataset: 'train' or 'test'.")
    parser.add_argument('index', type=int, help="Index number of the image in the dataset.")
    
    return parser.parse_args()

def show_image_and_label(dataset, index):
    """
    Displays the image and prints the corresponding label.

    Parameters:
    ----------
    dataset : str
        Either 'train' or 'test' to choose which dataset to use.
    index : int
        The index number of the image to be displayed.
    """
    # Create an instance of MnistData
    mnist_data = MnistData()

    # Load the dataset
    (train_images, train_labels), (test_images, test_labels) = mnist_data.load()

    # Choose the dataset based on the input argument
    if dataset == 'train':
        images, labels = train_images, train_labels
    else:
        images, labels = test_images, test_labels

    # Fetch the image and label at the given index
    image = images[index].reshape(28, 28)  # Reshape back to 28x28 for displaying
    label = np.argmax(labels[index])  # Convert one-hot encoded label back to a single digit

    # Show the image
    plt.imshow(image, cmap='gray')
    plt.title(f'Label: {label}')
    plt.show()

    # Print the label on the terminal
    print(f'The label for the image at index {index} is: {label}')

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    # Show the image and print the label
    show_image_and_label(args.dataset, args.index)
