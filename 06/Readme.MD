# MNIST Handwritten Digit Classification with Two-Layer Neural Network

This project implements a two-layer neural network for handwritten digit classification using the MNIST dataset. The model is trained using backpropagation and tested on custom handwritten digit images.

## Directory Structure
```
06/
├── Data/                     # Directory for storing custom data if needed
├── Model/                    # Directory for saving trained model pickle file
├── dataset/
│   ├── activations.py        # Contains the Activations class for various activation functions
│   ├── errors.py             # Contains the Errors class for error calculations
│   ├── layer.py              # Contains classes: Relu, Sigmoid, Affine, SoftmaxWithLoss
│   ├── mnist_data.py         # MNIST data loading class
├── module6.ipynb             # Jupyter Notebook for model training and validation
├── module6.py                # Script for testing the trained model on custom images
├── prediction_results.csv    # File to log prediction results
├── train.py                  # Script for training the two-layer network
├── two_layer_net_with_back_prop.py  # TwoLayerNetWithBackProp class definition
```

## Requirements
- Python 3.x
- `numpy`
- `matplotlib`
- `pickle`
- `argparse` (for handling command-line arguments in `module6.py`)
- `Pillow` or `OpenCV` (for image processing if needed for custom digit images)

### Installation
You can install the required libraries using:
```bash
pip install numpy matplotlib pillow argparse
```

## Files and Components

### 1. `layer.py`
Contains classes for backpropagation:
- `Relu`
- `Sigmoid`
- `Affine`
- `SoftmaxWithLoss`

### 2. `activations.py`
Implements the `Activations` class, providing activation functions.

### 3. `errors.py`
Contains the `Errors` class for error metrics like loss calculations.

### 4. `two_layer_net_with_back_prop.py`
Defines the `TwoLayerNetWithBackProp` class, implementing a two-layer neural network with backpropagation.

### 5. `train.py`
Trains the two-layer network on the MNIST dataset using `TwoLayerNetWithBackProp`.
- Hyperparameters:
  - Iterations: 10,000
  - Batch size: 16
  - Learning rate: 0.01
- The trained model is saved as `your_last_name_mnist_model.pkl` (replace `your_last_name` with your last name).

### 6. `module6.ipynb`
Jupyter Notebook demonstrating the training process, accuracy visualization, and testing of the trained model.
- Shows training steps with training and test accuracy for every epoch.
- Plots an accuracy graph over epochs.
- You can use `!python module6.py` in a notebook cell to run `module6.py` for testing.

### 7. `module6.py`
Tests the trained model on custom handwritten images.
- Arguments:
  - 1st argument: Image filename
  - 2nd argument: Expected digit in the image
- Example usage:
  ```bash
  python module6.py 3_2.png 3
  ```
- Output:
  - **Success**: `Success: Image 2_1.png for digit 2 is recognized as 2.`
  - **Failure**: `Fail: Image 2_1.png is for digit 2 but inference result is 3.`

### 8. `prediction_results.csv`
Logs prediction results from testing on custom images, indicating success or failure for each image.

## Training the Model
1. Run `train.py` to train the model using the MNIST dataset.
2. The trained model parameters are saved in the `Model` folder as `your_last_name_mnist_model.pkl`.

### Example Command
```bash
python train.py
```

## Testing the Model
1. Use `module6.py` to test the model on custom handwritten digit images.
2. Results are saved to `prediction_results.csv` and displayed in the terminal.

### Example Command
```bash
python module6.py data/3_2.png 3
```

## Results and Visualization
- **Training and Test Accuracy**: Printed in every epoch during training and shown in the accuracy plot.
- **Accuracy Graph**: Plotted in `module6.ipynb` to visualize the progress of training.

## Notes
- Ensure `iter_per_epoch` is an integer to prevent training issues.
- To execute scripts inside Jupyter Notebook, use:
  ```python
  !python <script_name.py>
  ```

