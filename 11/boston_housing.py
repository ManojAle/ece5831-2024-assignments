import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class BostonHousing:
    """
    A class to perform regression analysis on the Boston Housing dataset using a neural network.

    Methods:
        prepare_data(): Prepares and normalizes the Boston Housing dataset for training and testing.
        normalize_data(data): Normalizes the given dataset by subtracting the mean and dividing by the standard deviation.
        build_model(): Constructs the neural network model.
        train(epochs=20, batch_size=16): Trains the model using the training data.
        plot_loss(): Plots the training and validation loss over epochs.
        evaluate(): Evaluates the model on the test dataset and prints the test loss and mean absolute error (MAE).
    """

    def __init__(self):
        """
        Initializes the BostonHousing class with default values for model, history, training,
        and testing datasets.
        """
        self.model = None
        self.history = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.x_val = None
        self.y_val = None

    def prepare_data(self):
        """
        Prepares and normalizes the Boston Housing dataset.
        
        Splits the dataset into training, validation, and testing sets.
        Stores the normalized training and testing data and their corresponding labels.
        """
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data()
        self.x_train = self.normalize_data(x_train)
        self.x_test = self.normalize_data(x_test)
        self.y_train = y_train
        self.y_test = y_test
        self.x_val = self.x_train[:100]
        self.y_val = self.y_train[:100]
        self.x_train = self.x_train[100:]
        self.y_train = self.y_train[100:]

    def normalize_data(self, data):
        """
        Normalizes the given dataset.

        Args:
            data (numpy.ndarray): The dataset to normalize.

        Returns:
            numpy.ndarray: The normalized dataset.
        """
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

    def build_model(self):
        """
        Constructs the neural network model with two hidden layers and one output layer.

        The model uses ReLU activation for the hidden layers and outputs a single scalar value.
        """
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.x_train.shape[1],)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        self.model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

    def train(self, epochs=20, batch_size=16):
        """
        Trains the model using the training data.

        Args:
            epochs (int): Number of training epochs (default is 20).
            batch_size (int): Size of the training batch (default is 16).
        """
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)]
        self.history = self.model.fit(
            self.x_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.x_val, self.y_val),
            callbacks=callbacks
        )

    def plot_loss(self):
        """
        Plots the training and validation loss over epochs.

        This helps to visualize overfitting or underfitting during the training process.
        """
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, 'bo-', label='Training Loss')
        plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.show()

    def evaluate(self):
        """
        Evaluates the model on the test dataset.

        Prints the test loss and mean absolute error (MAE) for the predictions.

        Returns:
            None
        """
        results = self.model.evaluate(self.x_test, self.y_test)
        print(f"Test Loss: {results[0]}")
        print(f"Test MAE: {results[1]}")
