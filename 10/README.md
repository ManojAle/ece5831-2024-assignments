
# Dogs vs Cats Image Classification

## Description

This project implements an image classification pipeline to classify images of dogs and cats using a convolutional neural network (CNN). The dataset is prepared from the Kaggle Dogs vs Cats competition, and the code includes functionality for dataset preparation, model building, training, and inference.

## Features

- Dataset folder preparation for training, validation, and testing.
- Convolutional Neural Network (CNN) with data augmentation.
- Training with early stopping and model checkpoints.
- Visualization of training and validation accuracy/loss.
- Image prediction with class labels (Dog or Cat).

## Requirements

- Python 3.8+
- TensorFlow 2.9+
- Matplotlib
- pathlib
- shutil
- Kaggle Dogs vs Cats dataset

## Usage

### 1. Dataset Preparation

Download the dataset from the [Kaggle Dogs vs Cats competition](https://www.kaggle.com/competitions/dogs-vs-cats) and extract it to a directory named `dogs-vs-cats-original`.

Run the following to prepare dataset folders:

```python
from dogs_cats import DogsCats

dogs_cats = DogsCats()
dogs_cats.make_dataset_folders('train', 2400, 12000)
dogs_cats.make_dataset_folders('validation', 0, 2400)
dogs_cats.make_dataset_folders('test', 12000, 12500)
dogs_cats.make_dataset()
```

### 2. Build and Train the Model

```python
dogs_cats.build_network(augmentation=True)
dogs_cats.model.summary()
dogs_cats.train('model.dogs-cats.keras')
```

### 3. Load a Trained Model

```python
dogs_cats.load_model('model.dogs-cats.keras')
```

### 4. Make Predictions

```python
dogs_cats.predict('path/to/image.jpg')
```

## Results

```
Epoch 1/20
600/600 [==============================] - 253s 420ms/step - loss: 0.6820 - accuracy: 0.5619 - val_loss: 0.6387 - val_accuracy: 0.6369
Epoch 2/20
600/600 [==============================] - 247s 411ms/step - loss: 0.6185 - accuracy: 0.6584 - val_loss: 0.5523 - val_accuracy: 0.7175
Epoch 3/20
600/600 [==============================] - 253s 421ms/step - loss: 0.5567 - accuracy: 0.7148 - val_loss: 0.5131 - val_accuracy: 0.7500
Epoch 4/20
600/600 [==============================] - 261s 435ms/step - loss: 0.5222 - accuracy: 0.7400 - val_loss: 0.4760 - val_accuracy: 0.7738
Epoch 5/20
600/600 [==============================] - 266s 443ms/step - loss: 0.4954 - accuracy: 0.7609 - val_loss: 0.4445 - val_accuracy: 0.7906
Epoch 6/20
600/600 [==============================] - 273s 455ms/step - loss: 0.4722 - accuracy: 0.7778 - val_loss: 0.4245 - val_accuracy: 0.8017
Epoch 7/20
600/600 [==============================] - 259s 431ms/step - loss: 0.4534 - accuracy: 0.7826 - val_loss: 0.4075 - val_accuracy: 0.8171
Epoch 8/20
600/600 [==============================] - 273s 455ms/step - loss: 0.4362 - accuracy: 0.7968 - val_loss: 0.4238 - val_accuracy: 0.8131
Epoch 9/20
600/600 [==============================] - 261s 435ms/step - loss: 0.4155 - accuracy: 0.8087 - val_loss: 0.3831 - val_accuracy: 0.8313
Epoch 10/20
600/600 [==============================] - 264s 440ms/step - loss: 0.4018 - accuracy: 0.8158 - val_loss: 0.4192 - val_accuracy: 0.8142
Epoch 11/20
600/600 [==============================] - 271s 452ms/step - loss: 0.3945 - accuracy: 0.8190 - val_loss: 0.3944 - val_accuracy: 0.8240
Epoch 12/20
600/600 [==============================] - 260s 433ms/step - loss: 0.3783 - accuracy: 0.8283 - val_loss: 0.3935 - val_accuracy: 0.8275
```

## Author

Manoj Alexender


## Drive link for the Model

``` https://drive.google.com/file/d/1T-tCtTztw1jl6MiEjRWwckCchToegXPO/view?usp=sharing ```
