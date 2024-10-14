# Rock-Paper-Scissors Image Classification

## Description

This project provides two Python scripts for classifying images of rock, paper, and scissors using a trained model from Teachable Machine. The first script, `rock-paper-scissors.py`, takes a single image or a folder of images as input and classifies them into one of the three categories (rock, paper, or scissors). The second script, `rock-paper-scissors-live.py`, uses a webcam feed to classify hand gestures in real-time.

## Prerequisites

- Python 3.6+
- TensorFlow (for loading the Keras model)
- OpenCV (for live webcam feed in `rock-paper-scissors-live.py`)
- NumPy
- Pillow (for image loading and processing in `rock-paper-scissors.py`)

You can install the dependencies with the following command:

```bash
pip install tensorflow opencv-python numpy pillow
```

## Files

- `rock-paper-scissors.py`: A script to classify a single image or a folder of images as rock, paper, or scissors.
- `rock-paper-scissors-live.py`: A script that uses a webcam feed to classify hand gestures in real-time.
- `model/keras_model.h5`: The pre-trained model file (exported from Teachable Machine).
- `model/labels.txt`: The labels corresponding to the classes (exported from Teachable Machine).

## Usage

### 1. `rock-paper-scissors.py`

This script can be used to classify either a single image or a folder containing multiple images.

#### Example for Classifying a Single Image:

```bash
python rock-paper-scissors.py --image_path <path_to_image>
```

#### Example for Classifying All Images in a Folder:

```bash
python rock-paper-scissors.py --folder_path <path_to_folder>
```

#### Steps:

1. Place the `keras_model.h5` and `labels.txt` files inside the `model` directory.
2. Organize your samples in folders such as:
   - `Samples/Rock-samples`
   - `Samples/Scissors-samples`
   - `Samples/Paper-samples`
   
3. To classify a single image, run:
   ```bash
   python rock-paper-scissors.py --image_path Samples/Rock-samples/image1.jpg
   ```

4. To classify all images in a folder, run:
   ```bash
   python rock-paper-scissors.py --folder_path Samples/Rock-samples
   ```

5. The script will output the predicted class (rock, paper, or scissors) and confidence score for each image.

### 2. `rock-paper-scissors-live.py`

This script uses a live webcam feed to classify the hand gesture as rock, paper, or scissors in real-time.

#### Example:

```bash
python rock-paper-scissors-live.py
```

#### Steps:

1. Place the `keras_model.h5` and `labels.txt` files inside the `model` directory.
2. Run the script:
   ```bash
   python rock-paper-scissors-live.py
   ```
3. The webcam will start capturing frames, and the model will classify the hand gesture in real-time. The predicted class will be displayed on the screen.
4. Press the `q` key to exit the webcam feed.

## Directory Structure

```
rock-paper-scissors/
│
├── rock-paper-scissors.py        # Script for single image or folder batch classification
├── rock-paper-scissors-live.py   # Script for live classification using webcam
├── model/
│   ├── keras_model.h5            # Pre-trained model file (from Teachable Machine)
│   └── labels.txt                # Class labels (from Teachable Machine)
└── Samples/
    ├── Rock-samples/             # Folder containing rock sample images
    ├── Paper-samples/            # Folder containing paper sample images
    └── Scissors-samples/         # Folder containing scissors sample images
```

## Notes

- Ensure that `keras_model.h5` and `labels.txt` are located inside the `model` directory before running the scripts.
- The model file and labels file can be exported from Teachable Machine.

## License

This project is open-source and licensed under the MIT License.

