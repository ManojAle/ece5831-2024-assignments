
# HW5-03 Assignment

This repository contains files and scripts related to HW5-03, a project focused on handwritten digit recognition using a simple neural network model in Python. The project uses a pre-trained model to predict digits in custom handwritten images.

## Project Structure

```
HW5-03/
├── Data/                     # Contains folders for each digit (0-9) with handwritten images (e.g., Data/0/0_1.jpg)
├── Model/                    # Contains model weight files for the neural network (e.g., Model/sample_weight.pkl)
├── module5-3.py              # Script to load an image, preprocess it, and predict the digit using the neural network
├── module5-3.ipynb           # Jupyter Notebook to demonstrate testing functions and running predictions in a notebook environment
```

### Files and Folders

- **Data/**: This folder contains subdirectories for each digit (0-9), with each subdirectory holding multiple handwritten images. Images should be grayscale (28x28) with a white background and black digits.

- **Model/**: Contains the pre-trained model weights (`sample_weight.pkl`), which are loaded by `module5-3.py` to perform predictions.

- **module5-3.py**: A Python script that uses the `Mnist` class to load an image, preprocess it, and predict the digit. The script takes two command-line arguments:
  - `image_filename`: Path to the image file.
  - `digit`: The actual digit in the image, used to check prediction accuracy.

- **module5-3.ipynb**: A Jupyter Notebook that demonstrates the functionality of the `Mnist` class and shows predictions on multiple images. It also includes code to test individual functions before integrating them into the `Mnist` class.

## Usage

### Running `module5-3.py`

To run predictions on a specific image, use the following command:

```bash
python module5-3.py <image_filename> <digit>
```

Example:

```bash
python module5-3.py Data/3/3_2.jpg 3
```

This command will:
1. Load and preprocess the image.
2. Predict the digit in the image using the pre-trained neural network model.
3. Display the image and print whether the prediction was successful or not.

### Output

The output will be displayed in the terminal:
- **Success**: If the model's prediction matches the actual digit.
- **Fail**: If the model's prediction does not match the actual digit.

#### Example Output
```
Success: Image Data/3/3_2.jpg for digit 3 is recognized as 3.
```
or
```
Fail: Image Data/3/3_2.jpg is for digit 3 but the inference result is 5.
```

### Running Predictions on Multiple Images with `module5-3.ipynb`

The `module5-3.ipynb` notebook can be used to test multiple images automatically, load the network, and store results in a DataFrame. It is also useful for analyzing prediction results and visualizing accuracy across multiple samples.

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Matplotlib
- Pickle
- Pandas (for data analysis in the notebook)

**Note:** Create a `requirements.txt` file listing the dependencies if not already present.

## Author

Manoj Alexander

---

This `README.md` provides an overview of the project structure, usage, and example commands to run the prediction script. For more details on code implementation, refer to `module5-3.ipynb`.
```

### Explanation

- **Project Structure**: Describes each directory and file in `HW5-03`.
- **Usage**: Shows how to run `module5-3.py` with example commands.
- **Output**: Explains expected success/failure messages.
- **Requirements**: Lists necessary Python packages.
- **Installation**: Instructions to install dependencies.
  
