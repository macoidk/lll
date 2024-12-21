#SNNData_loader.py
from typing import Tuple
import numpy as np
from tensorflow.keras.datasets import cifar10


class DataLoader:
    @staticmethod
    def load_cifar10_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess CIFAR-10 dataset.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - x_train: Training data (input features)
                - y_train: Training labels
                - x_val: Validation data (input features)
                - y_val: Validation labels
                - x_test: Test data (input features)
                - y_test: Test labels
        """
        # Load CIFAR-10 data
        (x_train_full, y_train_full), (x_test, y_test) = cifar10.load_data()

        # Preprocess data
        x_train_full = x_train_full.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

        # One-hot encode labels
        y_train_full = np.eye(10)[y_train_full.squeeze()]
        y_test = np.eye(10)[y_test.squeeze()]

        # Split training set into train and validation
        num_training = int(len(x_train_full) * 0.7)
        num_validation = int(len(x_train_full) * 0.1)
        num_test = int(len(x_test) * 0.2)

        x_train = x_train_full[:num_training].reshape(num_training, -1).T
        y_train = y_train_full[:num_training].T

        x_val = x_train_full[num_training:num_training + num_validation].reshape(num_validation, -1).T
        y_val = y_train_full[num_training:num_training + num_validation].T

        x_test = x_test[:num_test].reshape(num_test, -1).T
        y_test = y_test[:num_test].T

        return x_train, y_train, x_val, y_val, x_test, y_test