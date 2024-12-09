
# Assignment: Binary, Multiclass Classification, and Regression with Deep Learning

This assignment implements binary classification, multiclass classification, and regression tasks using datasets provided by TensorFlow/Keras. The project is structured into modular Python classes and a Jupyter Notebook for demonstrating the workflows.

## Objective

The goal of this assignment is to practice the concepts from Chapter 4 of *Deep Learning with Python (2nd Edition)* by François Chollet. The tasks include:
1. **Binary Classification**: IMDB movie reviews sentiment analysis.
2. **Multiclass Classification**: Reuters newswire topic classification.
3. **Regression**: Boston Housing price prediction.

---

## Project Structure

```
.
├── imdb.py                # Binary Classification (IMDB Dataset)
├── reuters.py             # Multiclass Classification (Reuters Dataset)
├── boston_housing.py      # Regression (Boston Housing Dataset)
├── module11.ipynb         # Jupyter Notebook demonstrating all implementations
├── README.md              # This file
```

---

## Implementation Details

### 1. **Binary Classification: IMDB**

- **File**: `imdb.py`
- **Dataset**: IMDB dataset for binary sentiment classification (positive/negative).
- **Key Methods**:
  - `prepare_data()`: Prepares and vectorizes the dataset.
  - `build_model()`: Constructs a dense neural network.
  - `train()`: Trains the model with early stopping.
  - `plot_loss()`: Plots training and validation loss.
  - `plot_accuracy()`: Plots training and validation accuracy.
  - `evaluate()`: Evaluates the model on the test dataset, printing loss and accuracy.

---

### 2. **Multiclass Classification: Reuters**

- **File**: `reuters.py`
- **Dataset**: Reuters dataset for newswire topic classification (46 categories).
- **Key Methods**:
  - `prepare_data()`: Prepares and vectorizes the dataset. Converts labels to one-hot encoding.
  - `build_model()`: Constructs a dense neural network with softmax activation.
  - `train()`: Trains the model with early stopping.
  - `plot_loss()`: Plots training and validation loss.
  - `plot_accuracy()`: Plots training and validation accuracy.
  - `evaluate()`: Evaluates the model on the test dataset, printing loss and accuracy.

---

### 3. **Regression: Boston Housing**

- **File**: `boston_housing.py`
- **Dataset**: Boston Housing dataset for predicting house prices.
- **Key Methods**:
  - `prepare_data()`: Normalizes the dataset.
  - `build_model()`: Constructs a dense neural network for regression.
  - `train()`: Trains the model with early stopping.
  - `plot_loss()`: Plots training and validation loss.
  - `evaluate()`: Evaluates the model on the test dataset, printing loss and mean absolute error (MAE).

---

### 4. **Jupyter Notebook: Demonstrations**

- **File**: `module11.ipynb`
- **Purpose**: Demonstrates the implementation and results of the `Imdb`, `Reuters`, and `BostonHousing` classes.
- **Workflow**:
  1. Initialize each class.
  2. Prepare data for each task.
  3. Build, train, and evaluate the models.
  4. Plot the loss and accuracy/MAE.

---

## Installation

1. Clone the repository or download the files.
2. Install the required Python packages:
   ```bash
   pip install tensorflow matplotlib numpy
   ```
3. Navigate to the project directory.

---

## Usage

1. **Run the Jupyter Notebook**:
   Open `module11.ipynb` in Jupyter Notebook or any compatible environment and execute the cells to see the full implementation and results.

2. **Run Individual Scripts**:
   Each script (`imdb.py`, `reuters.py`, `boston_housing.py`) can be executed as a standalone program for testing purposes.

---

## Expected Output

- **IMDB Binary Classification**:
  - Training/Validation Loss and Accuracy plots.
  - Test Loss and Accuracy printed in the console.

- **Reuters Multiclass Classification**:
  - Training/Validation Loss and Accuracy plots.
  - Test Loss and Accuracy printed in the console.

- **Boston Housing Regression**:
  - Training/Validation Loss plot.
  - Test Loss and Mean Absolute Error (MAE) printed in the console.

---

## References

- *Deep Learning with Python (2nd Edition)* by François Chollet
- TensorFlow/Keras official documentation

---

## Notes

- Make sure your environment supports TensorFlow and Matplotlib.
- Modify hyperparameters such as `epochs` and `batch_size` in each class for experimentation.
- For additional customization, extend the classes as needed.
