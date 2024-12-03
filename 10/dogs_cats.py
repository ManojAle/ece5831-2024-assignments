import tensorflow as tf
from tensorflow.keras import layers
import os
import shutil
import pathlib
import matplotlib.pyplot as plt

class DogsCats:
    CLASS_NAMES = ['dog', 'cat']
    IMAGE_SHAPE = (180, 180, 3)
    BATCH_SIZE = 32
    BASE_DIR = pathlib.Path('dogs-vs-cats')
    SRC_DIR = pathlib.Path('dogs-vs-cats-original/train')

    def __init__(self):
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.model = None

    def make_dataset_folders(self, subset_name, start_index, end_index):
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
        return tf.keras.utils.image_dataset_from_directory(
            self.BASE_DIR / subset_name,
            image_size=self.IMAGE_SHAPE[:2],
            batch_size=self.BATCH_SIZE
        )

    def make_dataset(self):
        self.train_dataset = self._make_dataset('train')
        self.valid_dataset = self._make_dataset('validation')
        self.test_dataset = self._make_dataset('test')

    def build_network(self, augmentation=True):
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
        self.model = tf.keras.models.load_model(model_name)

    def predict(self, image_file):
        img = tf.keras.utils.load_img(image_file, target_size=self.IMAGE_SHAPE[:2])
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, axis=0)
        prediction = self.model.predict(img_array)[0][0]
        class_name = self.CLASS_NAMES[int(prediction > 0.5)]
        plt.imshow(img)
        plt.title(f"Prediction: {class_name}")
        plt.axis('off')
        plt.show()
