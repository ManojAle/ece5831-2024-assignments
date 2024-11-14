import os
import urllib.request
import gzip
import pickle
import numpy as np

class MnistData:
    """
    A class to download, process, and load the MNIST dataset for machine learning tasks.

    Attributes:
    ----------
    image_dim : tuple
        Dimensions of the MNIST images (28x28).
    image_size : int
        Total number of pixels in the MNIST images (28x28 = 784).
    dataset_dir : str
        Directory where the dataset will be downloaded and stored.
    dataset_pkl : str
        Name of the pickle file to store the processed dataset.
    url_base : str
        Base URL from where the MNIST dataset files will be downloaded.
    key_file : dict
        Mapping of dataset keys to their respective filenames for images and labels (train/test).

    Methods:
    -------
    _change_one_hot_label(y, num_class):
        Converts the label array into one-hot encoded format.
    _download(file_name):
        Downloads a specific file from the MNIST dataset if it does not already exist.
    _download_all():
        Downloads all MNIST dataset files (train/test images and labels).
    _load_images(file_name):
        Loads and reshapes image data from the specified gzip file.
    _load_labels(file_name):
        Loads label data from the specified gzip file.
    _create_dataset():
        Loads the dataset, processes it, and stores it as a pickle file.
    _init_dataset():
        Initializes the dataset by either loading the pickle file or downloading and processing the data.
    load():
        Returns normalized and one-hot encoded MNIST training and testing datasets.
    """

    image_dim = (28, 28)
    image_size = image_dim[0] * image_dim[1]
    dataset_dir = 'dataset'
    dataset_pkl = 'mnist.pkl'
    url_base = 'http://jrkwon.com/data/ece5831/mnist/'

    key_file = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }

    def __init__(self):
        """
        Initializes the MnistData object, creates the dataset directory if it does not exist,
        and either loads the existing dataset or downloads and processes it.
        """
        self.dataset = {}
        self.dataset_pkl_path = f'{self.dataset_dir}/{self.dataset_pkl}'

        # Create dataset_dir if the dir doesn't exist
        if os.path.exists(self.dataset_dir) is not True:
            os.mkdir(self.dataset_dir)

        self._init_dataset()

    def _change_one_hot_label(self, y, num_class):
        """
        Converts label array into one-hot encoding format.

        Parameters:
        y : numpy array
            The array of labels to be converted into one-hot format.
        num_class : int
            The number of classes (for MNIST, this is 10).
        """
        t = np.zeros((y.size, num_class))
        for idx, row in enumerate(t):
            row[y[idx]] = 1
        return t

    def _download(self, file_name):
        """
        Downloads a file from the specified URL if it does not already exist in the dataset directory.

        Parameters:
        file_name : str
            The name of the file to be downloaded.
        """
        file_path = self.dataset_dir + '/' + file_name

        if os.path.exists(file_path):
            print(f'File: {file_name} already exists.')
            return
        
        print(f'Downloading {file_name}...')

        # To resolve 406 Not Acceptable error
        opener = urllib.request.build_opener()
        opener.addheaders = [('Accept', '')]
        urllib.request.install_opener(opener)

        urllib.request.urlretrieve(self.url_base + file_name, file_path)
        print('Done')

    def _download_all(self):
        """
        Downloads all necessary MNIST dataset files (train/test images and labels).
        """
        for file_name in self.key_file.values():
            self._download(file_name)

    def _load_images(self, file_name):
        """
        Loads and reshapes image data from the specified gzip file.

        Parameters:
        file_name : str
            The name of the gzip file containing image data.

        Returns:
        images : numpy array
            The array of image data.
        """
        with gzip.open(file_name, 'rb') as f:
            images = np.frombuffer(f.read(), np.uint8, offset=16)
        images = images.reshape(-1, self.image_size)
        return images

    def _load_labels(self, file_name):
        """
        Loads label data from the specified gzip file.

        Parameters:
        file_name : str
            The name of the gzip file containing label data.

        Returns:
        labels : numpy array
            The array of label data.
        """
        with gzip.open(file_name, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
        return labels

    def _create_dataset(self):
        """
        Loads and processes the MNIST dataset, then stores it as a pickle file for future use.
        """
        file_name = f"{self.dataset_dir}/{self.key_file['train_images']}"
        self.dataset['train_images'] = self._load_images(file_name)

        file_name = f"{self.dataset_dir}/{self.key_file['train_labels']}"
        self.dataset['train_labels'] = self._load_labels(file_name)

        file_name = f"{self.dataset_dir}/{self.key_file['test_images']}"
        self.dataset['test_images'] = self._load_images(file_name)

        file_name = f"{self.dataset_dir}/{self.key_file['test_labels']}"
        self.dataset['test_labels'] = self._load_labels(file_name)

        with open(f'{self.dataset_pkl_path}', 'wb') as f:
            print(f'Pickle: {self.dataset_pkl_path} is being created.')
            pickle.dump(self.dataset, f)
            print('Done.')

    def _init_dataset(self):
        """
        Initializes the dataset by either loading from an existing pickle file or downloading and processing it.
        """
        self._download_all()
        if os.path.exists(f'{self.dataset_pkl_path}'):
            with open(f'{self.dataset_pkl_path}', 'rb') as f:
                print(f'Pickle: {self.dataset_pkl_path} already exists.')
                print('Loading...')
                self.dataset = pickle.load(f)
                print('Done.')
        else:
            self._create_dataset()

    def load(self):
        """
        Loads the normalized and one-hot encoded MNIST dataset.

        Returns:
        tuple:
            A tuple containing the training images and labels, and the test images and labels.
        """
        # Normalize image datasets
        for key in ('train_images', 'test_images'):
            self.dataset[key] = self.dataset[key].astype(np.float32)
            self.dataset[key] /= 255.0

        # One-hot encoding for labels
        for key in ('train_labels', 'test_labels'):
            self.dataset[key] = self._change_one_hot_label(self.dataset[key], 10)

        return (self.dataset['train_images'], self.dataset['train_labels']), \
               (self.dataset['test_images'], self.dataset['test_labels'])


if __name__ == "__main__":
    mnist_data = MnistData()
    (train_images, train_labels), (test_images, test_labels) = mnist_data.load()

    # Output information about the dataset
    print(f"Training set size: {train_images.shape[0]} images")
    print(f"Test set size: {test_images.shape[0]} images")
    print(f"Each image is flattened into: {train_images.shape[1]} pixels")
    print(f"Sample label (one-hot encoded): {train_labels[0]}")
    print(f"Sample test image label (one-hot encoded): {test_labels[0]}")
