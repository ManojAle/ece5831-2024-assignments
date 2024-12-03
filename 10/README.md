
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

Ensure the accuracy is greater than 0.7 for the model.

## Author

Manoj Alexender
