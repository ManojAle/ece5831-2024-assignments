### README for LeNet CNN Implementation

#### Project Structure
```
Project/
│
├── Data/                   # Folder containing dataset-related files (if applicable)
│
├── Model/                  # Folder to store the trained LeNet models
│   └── manoj.h5.keras 
│
├── le_net.py               # Python file containing the implementation of the LeNet CNN class
│
├── module8.ipynb           # Jupyter Notebook for testing and visualizing predictions
│
└── module8.py              # Python script for testing the LeNet class
```

---

#### Description

This project implements the classic **LeNet-5 Convolutional Neural Network (CNN)** for digit classification using the **MNIST dataset**. The model architecture includes convolutional layers, average pooling, and fully connected layers designed for classifying grayscale 28x28 images into one of 10 digit classes (0–9).

---

#### Files and Components

1. **`le_net.py`**:
   - Defines the `LeNet` class.
   - Includes methods to:
     - Build and compile the LeNet model.
     - Train the model on the MNIST dataset.
     - Save and load trained models.
     - Predict class labels for input images.

2. **`module8.ipynb`**:
   - A Jupyter Notebook for interactive testing.
   - Visualizes predictions and results for a set of test images.

3. **`module8.py`**:
   - A standalone script for testing the `LeNet` class.
   - Accepts an input image and the expected digit via command-line arguments.
   - Processes the image and predicts the digit class.
   - Provides success/failure feedback based on the prediction.

4. **`Model/`**:
   - Directory to save and load trained `.keras` model files.

---

#### Usage

1. **Training the Model**:
   To train the LeNet model on the MNIST dataset:
   ```python
   from le_net import LeNet

   # Initialize and train the model
   lenet = LeNet(batch_size=64, epochs=10)
   lenet.train()

   # Save the trained model
   lenet.save("Model/alexender_cnn_model")
   ```

2. **Loading a Pre-trained Model**:
   To load a previously trained model:
   ```python
   from le_net import LeNet

   # Initialize the LeNet class and load the model
   lenet = LeNet()
   lenet.load("Model/alexender_cnn_model")
   ```

3. **Prediction**:
   To predict the class label for a test image:
   ```python
   # Example: Predicting for a single preprocessed image
   predicted_digit = lenet.predict(preprocessed_image)
   print(f"Predicted digit: {predicted_digit[0]}")
   ```

4. **Running `module8.py`**:
   Test a saved LeNet model with an image:
   ```bash
   python module8.py <path_to_image> <expected_digit>
   ```

   Example:
   ```bash
   python module8.py test_image.png 5
   ```

5. **Interactive Testing (`module8.ipynb`)**:
   Use the provided Jupyter Notebook to visualize predictions and test the model on multiple images interactively.

---

#### Requirements

1. **Python Packages**:
   - `tensorflow`
   - `numpy`
   - `matplotlib`
   - `opencv-python`

   Install dependencies using:
   ```bash
   pip install tensorflow numpy matplotlib opencv-python
   ```

2. **Dataset**:
   - The MNIST dataset is loaded automatically using TensorFlow's `keras.datasets` module.

---

#### Model Details

The LeNet CNN architecture:
- **Input**: Grayscale image (28x28x1).
- **Layers**:
  - `Conv2D`: 6 filters, kernel size 5x5, sigmoid activation.
  - `AveragePooling2D`: Pool size 2x2.
  - `Conv2D`: 16 filters, kernel size 5x5, sigmoid activation.
  - `AveragePooling2D`: Pool size 2x2.
  - `Flatten`
  - Fully connected `Dense` layers (120, 84 neurons).
  - Output `Dense` layer with 10 neurons (softmax activation).

---

#### Example Workflow

1. Train the model:
   ```python
   from le_net import LeNet
   lenet = LeNet(batch_size=64, epochs=10)
   lenet.train()
   lenet.save("Model/alexender_cnn_model")
   ```

2. Test the model using the Python script:
   ```bash
   python module8.py test_image.png 5
   ```

3. Visualize results in `module8.ipynb`.