from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np

class LeNet:
    """
    A class to represent the LeNet Convolutional Neural Network (CNN) for digit classification on the MNIST dataset.
    
    Attributes:
        batch_size (int): Batch size for training the model.
        epochs (int): Number of epochs for training the model.
        model (Sequential): The Keras Sequential model instance.
    """
    def __init__(self, batch_size=32, epochs=20):
        """
        Initializes the LeNet instance, setting the batch size, number of epochs, and building the model.

        Args:
            batch_size (int, optional): Batch size for training. Defaults to 32.
            epochs (int, optional): Number of training epochs. Defaults to 20.
        """
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self._create_lenet()
        self._compile()

    def _create_lenet(self):
        """
        Creates the LeNet CNN architecture and initializes the model.
        The architecture includes:
            - Convolutional layers
            - Average Pooling layers
            - Fully connected layers
            - A softmax output layer
        """
        self.model = Sequential([
            Conv2D(filters=6, kernel_size=(5, 5), activation='sigmoid', input_shape=(28, 28, 1), padding='same'),
            AveragePooling2D(pool_size=(2, 2), strides=2),
            Conv2D(filters=16, kernel_size=(5, 5), activation='sigmoid', padding='same'),
            AveragePooling2D(pool_size=(2, 2), strides=2),
            Flatten(),
            Dense(120, activation='sigmoid'),
            Dense(84, activation='sigmoid'),
            Dense(10, activation='softmax')
        ])
    
    def _compile(self):
        """
        Compiles the LeNet model with the Adam optimizer, categorical crossentropy loss, and accuracy metric.
        Raises:
            ValueError: If the model is not initialized before compilation.
        """
        if self.model is None:
            raise ValueError('Error: Create a model first.')
        self.model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def _preprocess(self):
        """
        Preprocesses the MNIST dataset for training and testing:
            - Normalizes pixel values to the range [0, 1].
            - Reshapes images to match the input shape of the CNN (28, 28, 1).
            - One-hot encodes the labels for classification.
        """
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # Normalize pixel values to range [0, 1]
        x_train = x_train / 255.0
        x_test = x_test / 255.0

        # Reshape to match input shape of CNN
        self.x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        self.x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

        # One-hot encode the labels
        self.y_train = to_categorical(y_train, 10)
        self.y_test = to_categorical(y_test, 10)

    def train(self):
        """
        Trains the LeNet model on the preprocessed MNIST dataset.
        """
        self._preprocess()
        self.model.fit(self.x_train, self.y_train, batch_size=self.batch_size, epochs=self.epochs, validation_data=(self.x_test, self.y_test))

    def save(self, model_path_name):
        """
        Saves the trained model to a file.

        Args:
            model_path_name (str): The base path and name for saving the model. 
                                   The `.keras` extension is added automatically.
        """
        self.model.save(model_path_name + ".keras")
        print(f"Model saved as {model_path_name}.keras")

    def load(self, model_path_name):
        """
        Loads a trained model from a `.keras` file.

        Args:
            model_path_name (str): The base path and name of the model file to load (without `.keras` extension).
        """
        self.model = load_model(model_path_name + ".keras")
        print(f"Model loaded from {model_path_name}.keras")

    def predict(self, images):
        """
        Predicts the labels for a list or batch of images using the trained model.

        Args:
            images (ndarray): Input images, either a single image with shape (28, 28, 1) or a batch with shape (n_samples, 28, 28, 1).
        
        Returns:
            ndarray: Predicted class labels for the input images.

        Raises:
            ValueError: If the input images do not have the correct shape.
        """
        # Ensure the images are numpy arrays with the correct shape
        if len(images.shape) == 3:  # Single image without batch dimension
            images = np.expand_dims(images, axis=0)
        elif len(images.shape) != 4:
            raise ValueError("Input images must have shape (n_samples, 28, 28, 1) or (28, 28, 1) for a single image.")
        
        predictions = self.model.predict(images)
        predicted_classes = np.argmax(predictions, axis=1)
        return predicted_classes
