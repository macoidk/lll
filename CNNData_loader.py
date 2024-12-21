from typing import Tuple
import numpy as np
from tensorflow.keras.datasets import cifar10


class DataLoader:
    @staticmethod
    def load_cifar10_data(total_images: int) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess CIFAR-10 dataset with user-specified total number of images.
        Split ratio: 70% training, 10% validation, 20% test

        Args:
            total_images (int): Total number of images to use from the dataset

        Returns:
            Tuple containing training, validation, and test data with their labels
        """
        # Load CIFAR-10 data
        (x_train_full, y_train_full), (x_test_full, y_test_full) = cifar10.load_data()

        # Calculate split sizes based on total_images
        num_train = int(total_images * 0.7)  # 70% for training
        num_val = int(total_images * 0.1)  # 10% for validation
        num_test = total_images - num_train - num_val  # Remaining for test

        # Take sequential samples instead of random
        x_data = np.concatenate([x_train_full[:num_train + num_val], x_test_full[:num_test]])
        y_data = np.concatenate([y_train_full[:num_train + num_val], y_test_full[:num_test]])

        # Preprocess data - normalize to [0,1]
        x_data = x_data.astype('float32') / 255.0

        # Convert from (N,H,W,C) to (N,C,H,W) format
        x_data = np.transpose(x_data, (0, 3, 1, 2))

        # One-hot encode labels
        y_data = np.eye(10)[y_data.squeeze()]

        # Split the data
        x_train = x_data[:num_train]
        y_train = y_data[:num_train].T

        x_val = x_data[num_train:num_train + num_val]
        y_val = y_data[num_train:num_train + num_val].T

        x_test = x_data[num_train + num_val:]
        y_test = y_data[num_train + num_val:].T

        # Print dataset sizes
        print(f"\nDataset Split Summary:")
        print(f"Training set: {x_train.shape[0]} images")
        print(f"Validation set: {x_val.shape[0]} images")
        print(f"Test set: {x_test.shape[0]} images")
        print(f"Total: {x_train.shape[0] + x_val.shape[0] + x_test.shape[0]} images")

        return x_train, y_train, x_val, y_val, x_test, y_test