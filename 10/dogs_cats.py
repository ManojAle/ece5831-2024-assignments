import tensorflow as tf
from tensorflow.keras import layers
import os
import shutil
import pathlib
import matplotlib.pyplot as plt

class DogsCats:
    """
    A class to handle the Dogs vs. Cats image classification task.

    Attributes:
        CLASS_NAMES (list): List of class labels ['dog', 'cat'].
        IMAGE_SHAPE (tuple): Shape of input images (height, width, channels).
        BATCH_SIZE (int): Batch size for training and validation datasets.
        BASE_DIR (pathlib.Path): Base directory for dataset folders.
        SRC_DIR (pathlib.Path): Source directory for raw dataset images.

    Methods:
        __init__(): Initializes train_dataset, valid_dataset, test_dataset, and model as None.
        make_dataset_folders(subset_name, start_index, end_index): Creates folder structure for the dataset.
        _make_dataset(subset_name): Creates a TensorFlow Dataset for a specified subset.
        make_dataset(): Generates TensorFlow Datasets for training, validation, and testing.
        build_network(augmentation=True): Builds and compiles a convolutional neural network.
        train(model_name): Trains the model and plots training/validation accuracy and loss.
        load_model(model_name): Loads a pre-trained model from a file.
        predict(image_file): Makes predictions on a given image and displays the result.
    """
    
    CLASS_NAMES = ['dog', 'cat']
    IMAGE_SHAPE = (180, 180, 3)
    BATCH_SIZE = 32
    BASE_DIR = pathlib.Path('dogs-vs-cats')
    SRC_DIR = pathlib.Path('dogs-vs-cats-original/train')

    def __init__(self):
        """Initializes the DogsCats class with dataset and model attributes as None."""
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.model = None

    def make_dataset_folders(self, subset_name, start_index, end_index):
        """
        Creates a folder structure for the dataset.

        Args:
            subset_name (str): Name of the subset (train, validation, or test).
            start_index (int): Starting index for image file selection.
            end_index (int): Ending index for image file selection.
        """
        for category in self.CLASS_NAMES:
            subset_dir = self.BASE_DIR / subset_name / category
            if not os.path.exists(subset_dir):
                os.makedirs(subset_dir)
            files = [f'{category}.{i}.jpg' for i in range(start_index, end_index)]
            for i, file in enumerate(files):
                src = self.SRC_DIR / file
                dst = subset_dir / file
                if os.path.exists(src):
                    shutil.copyfile(src, dst)
                if i % 100 == 0:  # Print status every 100 files
                    print(f'{src} => {dst}')

    def _make_dataset(self, subset_name):
        """
        Creates a TensorFlow Dataset for a specified subset.

        Args:
            subset_name (str): Name of the subset (train, validation, or test).

        Returns:
            tf.data.Dataset: A TensorFlow Dataset object for the subset.
        """
        return tf.keras.utils.image_dataset_from_directory(
            self.BASE_DIR / subset_name,
            image_size=self.IMAGE_SHAPE[:2],
            batch_size=self.BATCH_SIZE
        )

    def make_dataset(self):
        """
        Generates TensorFlow Datasets for training, validation, and testing.
        """
        self.train_dataset = self._make_dataset('train')
        self.valid_dataset = self._make_dataset('validation')
        self.test_dataset = self._make_dataset('test')

    def build_network(self, augmentation=True):
        """
        Builds and compiles a convolutional neural network.

        Args:
            augmentation (bool): Whether to include data augmentation layers. Default is True.
        """
        inputs = layers.Input(shape=self.IMAGE_SHAPE)
        if augmentation:
            data_augmentation = tf.keras.Sequential([
                layers.RandomFlip('horizontal'),
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.2),
            ])
            x = data_augmentation(inputs)
        else:
            x = inputs
        x = layers.Rescaling(1./255)(x)
        x = layers.Conv2D(32, kernel_size=3, activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(64, kernel_size=3, activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(128, kernel_size=3, activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation='relu')(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)

        self.model = tf.keras.Model(inputs, outputs)
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    def train(self, model_name):
        """
        Trains the model and plots training/validation accuracy and loss.

        Args:
            model_name (str): File name to save the trained model.
        """
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=3),
            tf.keras.callbacks.ModelCheckpoint(filepath=model_name),
            tf.keras.callbacks.TensorBoard(log_dir='./logs')
        ]
        history = self.model.fit(
            self.train_dataset,
            validation_data=self.valid_dataset,
            epochs=20,
            callbacks=callbacks
        )
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        plt.figure()
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy')

        plt.figure()
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.show()

    def load_model(self, model_name):
        """
        Loads a pre-trained model from a file.

        Args:
            model_name (str): Path to the model file.
        """
        self.model = tf.keras.models.load_model(model_name)

    def predict(self, image_file):
        """
        Makes predictions on a given image and displays the result.

        Args:
            image_file (str): Path to the image file.
        """
        img = tf.keras.utils.load_img(image_file, target_size=self.IMAGE_SHAPE[:2])
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, axis=0)
        prediction = self.model.predict(img_array)[0][0]
        class_name = self.CLASS_NAMES[int(prediction > 0.5)]
        plt.imshow(img)
        plt.title(f"Prediction: {class_name}")
        plt.axis('off')
        plt.show()
