import json
import numpy as np
from pathlib import Path

class AutoSpeechDataset:
    def __init__(self, dataset_dir):
        """
        AutoSpeechDataset class to load and manage dataset files.

        Parameters:
        dataset_dir (str): Path to the dataset directory.

        Attributes:
        - train_dataset: List of training data samples.
        - train_label: Numpy array of labels.
        - test_dataset: List of test data samples.
        - metadata: Dictionary containing dataset metadata.
        """
        self.dataset_dir = Path(dataset_dir)
        self.metadata = self._read_metadata(self.dataset_dir / "meta.json")
        
        # Initialize dataset variables
        self.train_dataset = None
        self.train_label = None
        self.test_dataset = None

    def load_data(self):
        """Loads training and test datasets."""
        self.train_dataset = self._read_dataset(self.dataset_dir / "train.pkl")
        self.train_label = self._read_label(self.dataset_dir / "train.solution")
        self.test_dataset = self._read_dataset(self.dataset_dir / "test.pkl")

    def get_train(self):
        return self.train_dataset, self.train_label

    def get_test(self):
        return self.test_dataset

    def get_metadata(self):
        return self.metadata

    def _read_metadata(self, metadata_path):
        """Reads dataset metadata from a JSON file."""
        try:
            with open(metadata_path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(f"Error reading metadata: {e}")

    def _read_dataset(self, dataset_path):
        """Reads dataset from a pickle (.pkl) file."""
        try:
            with open(dataset_path, "rb") as f:
                return pickle.load(f)
        except (FileNotFoundError, pickle.UnpicklingError) as e:
            raise ValueError(f"Error reading dataset: {e}")

    def _read_label(self, label_path):
        """Reads label data, assuming it's a text file."""
        try:
            return np.loadtxt(label_path)
        except (FileNotFoundError, ValueError) as e:
            raise ValueError(f"Error reading label file: {e}")

    def get_class_num(self):
        """Returns the number of classes in the dataset."""
        return self.metadata.get("class_num", None)

    def get_train_num(self):
        """Returns the number of training instances."""
        return self.metadata.get("train_num", None)

    def get_test_num(self):
        """Returns the number of test instances."""
        return self.metadata.get("test_num", None)

    def get_language(self):
        """Returns dataset language (ZH or EN)."""
        return self.metadata.get("language", None)