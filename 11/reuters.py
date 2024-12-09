import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class Reuters:
    """
    A class to perform multiclass classification on the Reuters dataset using a neural network.

    Methods:
        prepare_data(num_words=10000): Prepares and vectorizes the Reuters dataset for training and testing.
        vectorize_sequences(sequences, dimension=10000): Converts sequences of integers into one-hot encoded vectors.
        build_model(): Constructs the neural network model.
        train(epochs=20, batch_size=512): Trains the model using the training data.
        plot_loss(): Plots the training and validation loss over epochs.
        plot_accuracy(): Plots the training and validation accuracy over epochs.
        evaluate(): Evaluates the model on the test dataset and prints the test loss and accuracy.
    """

    def __init__(self):
        """
        Initializes the Reuters class with default values for model, history, training,
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

    def prepare_data(self, num_words=10000):
        """
        Prepares and vectorizes the Reuters dataset for multiclass classification.

        Args:
            num_words (int): The maximum number of words to include in the vocabulary (default is 10,000).

        Splits the dataset into training, validation, and testing sets.
        Converts sequences of integers into one-hot encoded vectors for input and one-hot encoded labels.
        """
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.reuters.load_data(num_words=num_words)
        self.x_train = self.vectorize_sequences(x_train)
        self.x_test = self.vectorize_sequences(x_test)
        self.y_train = tf.keras.utils.to_categorical(y_train)
        self.y_test = tf.keras.utils.to_categorical(y_test)
        self.x_val = self.x_train[:1000]
        self.y_val = self.y_train[:1000]
        self.x_train = self.x_train[1000:]
        self.y_train = self.y_train[1000:]

    def vectorize_sequences(self, sequences, dimension=10000):
        """
        Converts sequences of integers into one-hot encoded vectors.

        Args:
            sequences (list of lists): A list of integer sequences to be vectorized.
            dimension (int): The dimension of the one-hot encoded vectors (default is 10,000).

        Returns:
            numpy.ndarray: A 2D array where each row is the one-hot encoded representation of a sequence.
        """
        results = np.zeros((len(sequences), dimension))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.0
        return results

    def build_model(self):
        """
        Constructs the neural network model with two hidden layers and a softmax output layer.

        The model uses ReLU activation for the hidden layers and softmax activation for multiclass classification.
        """
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(46, activation='softmax')
        ])
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, epochs=20, batch_size=512):
        """
        Trains the model using the training data.

        Args:
            epochs (int): Number of training epochs (default is 20).
            batch_size (int): Size of the training batch (default is 512).
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

    def plot_accuracy(self):
        """
        Plots the training and validation accuracy over epochs.

        This helps to monitor how well the model generalizes to unseen data.
        """
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, 'bo-', label='Training Accuracy')
        plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.show()

    def evaluate(self):
        """
        Evaluates the model on the test dataset.

        Prints the test loss and accuracy for multiclass classification.

        Returns:
            None
        """
        results = self.model.evaluate(self.x_test, self.y_test)
        print(f"Test Loss: {results[0]}")
        print(f"Test Accuracy: {results[1]}")
