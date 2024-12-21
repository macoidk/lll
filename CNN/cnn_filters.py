#model 3
#conv 3
#accuracy = 0.57
#time to train =
#filters initialization -

#Optimizer Adam

import numpy as np
from Layers import Conv2d, Linear, Flatten, MaxPool2d
from BatchNorm2d import BatchNorm2d
from LossFunctions import CrossEntropyLoss
from Optimizers import Adam
from CNNData_loader import DataLoader
from Visualizer import Visualizer, FilterVisualizer
from LossFunctions import L2RegularizationLoss
from Filters import get_filters_for_layer
from DataAugmentation import HorizontalFlip, RandomRotation, RandomBrightness, Compose
import matplotlib.pyplot as plt
import os

class CNN:
    def __init__(self):
        self.conv1 = Conv2d(in_channels=3, out_channels=23, kernel_size=3, padding=1)
        self.conv1.W = get_filters_for_layer(0, ['vertical_edges', 'horizontal_edges', 'diagonal1_edges',
                                                 'diagonal2_edges', 'emboss', 'sharpen', 'blur_gaussian', 'motion_blur',
                                                 'sobel_x', 'sobel_y', 'prewitt_x', 'prewitt_y', 'laplacian', 'unsharp_mask',
                                                 'median_blur', 'bilateral_blur', 'wiener_filter', 'line_filter_0_degrees',
                                                  'line_filter_45_degrees', 'line_filter_90_degrees', 'line_filter_135_degrees', 'hog_filter',
                                                  'color_histogram'])
        self.bn1 = BatchNorm2d(23)
        self.pool1 = MaxPool2d(kernel_size=2, stride=2)
#16x16
        self.conv2 = Conv2d(in_channels=23, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = BatchNorm2d(32)
        self.pool2 = MaxPool2d(kernel_size=2, stride=2)
#8x8
        self.conv3 = Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = BatchNorm2d(64)
        self.pool3 = MaxPool2d(kernel_size=2, stride=2)
#4x4
        self.conv4 = Conv2d(in_channels=64, out_channels=104, kernel_size=3, padding=1)
        self.bn4 = BatchNorm2d(104)
        self.pool4 = MaxPool2d(kernel_size=2, stride=2)

#2x2
        self.conv5 = Conv2d(in_channels=104, out_channels=256, kernel_size=3, padding=1)
        self.bn5 = BatchNorm2d(256)


        self.flatten = Flatten()
        self.fc1 = Linear(in_features=256 * 2 * 2, out_features=512)
        self.fc2 = Linear(in_features=512, out_features=10)

        # Loss functions and optimizer
        self.criterion = CrossEntropyLoss()
        self.l2_reg = L2RegularizationLoss()
        self.optimizer = Adam(beta1=0.9, beta2=0.999)

        self.lambda_reg = 0.0001 # L2 regularization strength 0.0001

        # Initialize optimizer
        self.parameters = {
            'conv1_w': self.conv1.W, 'conv1_b': self.conv1.b,
            'conv2_w': self.conv2.W, 'conv2_b': self.conv2.b,
            'conv3_w': self.conv3.W, 'conv3_b': self.conv3.b,
            'conv4_w': self.conv4.W, 'conv4_b': self.conv4.b,
            'conv5_w': self.conv5.W, 'conv5_b': self.conv5.b,
            'fc1_w': self.fc1.W, 'fc1_b': self.fc1.b,
            'fc2_w': self.fc2.W, 'fc2_b': self.fc2.b,
            'bn1_gamma': self.bn1.gamma, 'bn1_beta': self.bn1.beta,
            'bn2_gamma': self.bn2.gamma, 'bn2_beta': self.bn2.beta,
            'bn3_gamma': self.bn3.gamma, 'bn3_beta': self.bn3.beta,
            'bn4_gamma': self.bn4.gamma, 'bn4_beta': self.bn4.beta,
            'bn5_gamma': self.bn5.gamma, 'bn5_beta': self.bn5.beta
        }

        self.optimizer.initialize(self.parameters)

    def forward(self, x, training=True):
        self.activations = {}

        # Normalize input
        x = (x - np.mean(x, axis=(0, 2, 3), keepdims=True)) / (np.std(x, axis=(0, 2, 3), keepdims=True) + 1e-5)

        # First conv block
        x = self.conv1.forward(x)
        self.activations['conv1'] = x
        x = self.bn1.forward(x, training)
        self.activations['bn1'] = x
        x = np.maximum(0, x)  # ReLU
        self.activations['relu1'] = x
        x = self.pool1.forward(x)
        self.activations['pool1'] = x

        # Second conv block
        x = self.conv2.forward(x)
        self.activations['conv2'] = x
        x = self.bn2.forward(x, training)
        self.activations['bn2'] = x
        x = np.maximum(0, x)  # ReLU
        self.activations['relu2'] = x
        x = self.pool2.forward(x)
        self.activations['pool2'] = x

        # Third conv block
        x = self.conv3.forward(x)
        self.activations['conv3'] = x
        x = self.bn3.forward(x, training)
        self.activations['bn3'] = x
        x = np.maximum(0, x)  # ReLU
        self.activations['relu3'] = x
        x = self.pool3.forward(x)
        self.activations['pool3'] = x

        x = self.conv4.forward(x)
        self.activations['conv4'] = x
        x = self.bn4.forward(x, training)
        self.activations['bn4'] = x
        x = np.maximum(0, x)  # ReLU
        self.activations['relu4'] = x
        x = self.pool4.forward(x)
        self.activations['pool4'] = x

        x = self.conv5.forward(x)
        self.activations['conv5'] = x
        x = self.bn5.forward(x, training)
        self.activations['bn5'] = x
        x = np.maximum(0, x)  # ReLU
        self.activations['relu5'] = x

        self.activations['pool5'] = x

        # Fully connected layers
        x = self.flatten.forward(x)
        self.activations['flatten'] = x
        x = x.T
        x = self.fc1.forward(x)
        self.activations['fc1'] = x
        x = np.maximum(0, x)  # ReLU
        self.activations['relu6'] = x

        x = self.fc2.forward(x)
        self.activations['fc2'] = x

        # Softmax with numerical stability
        x = x - np.max(x, axis=0, keepdims=True)  # For numerical stability
        exp_scores = np.exp(x)
        probs = exp_scores / np.sum(exp_scores, axis=0, keepdims=True)
        self.activations['softmax'] = probs
        return probs

    def compute_loss(self, output, y):
        """Compute loss with L2 regularization"""
        ce_loss = self.criterion.forward(output, y)

        weights = {
            'conv1_w': self.conv1.W,
            'conv2_w': self.conv2.W,
            'conv3_w': self.conv3.W,
            'conv4_w': self.conv4.W,
            'conv5_w': self.conv5.W,
            'fc1_w': self.fc1.W,
            'fc2_w': self.fc2.W
        }

        reg_loss = self.l2_reg.forward(weights, self.lambda_reg)
        return ce_loss + reg_loss

    def backward(self, x, y):
        # Compute gradients
        dout = self.criterion.backward(self.activations['softmax'], y)

        # Backward pass through the network
        dout = self.fc2.backward(dout)

        # ReLU backward for fc1
        dout = dout * (self.activations['relu6'] > 0)
        dout = self.fc1.backward(dout)

        # Reshape for convolution layers
        dout = self.flatten.backward(dout)


        dout = dout * (self.activations['relu5'] > 0)
        dout = self.bn5.backward(dout)[0]
        dout = self.conv5.backward(dout)

        dout = self.pool4.backward(dout)
        dout = dout * (self.activations['relu4'] > 0)
        dout = self.bn4.backward(dout)[0]
        dout = self.conv4.backward(dout)

        # Third conv block backward
        dout = self.pool3.backward(dout)
        dout = dout * (self.activations['relu3'] > 0)  # ReLU
        dout = self.bn3.backward(dout)[0]
        dout = self.conv3.backward(dout)

        # Second conv block backward
        dout = self.pool2.backward(dout)
        dout = dout * (self.activations['relu2'] > 0)  # ReLU
        dout = self.bn2.backward(dout)[0]
        dout = self.conv2.backward(dout)

        # First conv block backward
        dout = self.pool1.backward(dout)
        dout = dout * (self.activations['relu1'] > 0)  # ReLU
        dx1, dgamma1, dbeta1 = self.bn1.backward(dout)
        dout = self.conv1.backward(dx1)

        # Calculate L2 regularization gradients
        gradients = {
            'conv1_w': self.conv1.dW + self.l2_reg.backward(self.conv1.W, self.lambda_reg),
            'conv1_b': self.conv1.db,
            'conv2_w': self.conv2.dW + self.l2_reg.backward(self.conv2.W, self.lambda_reg),
            'conv2_b': self.conv2.db,
            'conv3_w': self.conv3.dW + self.l2_reg.backward(self.conv3.W, self.lambda_reg),
            'conv3_b': self.conv3.db,
            'conv4_w': self.conv4.dW + self.l2_reg.backward(self.conv4.W, self.lambda_reg),
            'conv4_b': self.conv4.db,
            'conv5_w': self.conv5.dW + self.l2_reg.backward(self.conv5.W, self.lambda_reg),
            'conv5_b': self.conv5.db,
            'fc1_w': self.fc1.dW + self.l2_reg.backward(self.fc1.W, self.lambda_reg),
            'fc1_b': self.fc1.db,
            'fc2_w': self.fc2.dW + self.l2_reg.backward(self.fc2.W, self.lambda_reg),
            'fc2_b': self.fc2.db,
            'bn1_gamma': dgamma1,
            'bn1_beta': dbeta1,
            'bn2_gamma': self.bn2.cache['dgamma'],
            'bn2_beta': self.bn2.cache['dbeta'],
            'bn3_gamma': self.bn3.cache['dgamma'],
            'bn3_beta': self.bn3.cache['dbeta'],
            'bn4_gamma': self.bn4.cache['dgamma'],
            'bn4_beta': self.bn4.cache['dbeta'],
            'bn5_gamma': self.bn5.cache['dgamma'],
            'bn5_beta': self.bn5.cache['dbeta']
        }

        return gradients

    def update_parameters(self, gradients, learning_rate):
        """Update parameters using optimizer"""
        self.optimizer.update(gradients, learning_rate)



    def visualize_network_filters(self, save_path=None):
        """
        Visualize filters from all convolutional layers of the network

        Args:
            save_path (str, optional): Path to save the visualization plot
        """
        FilterVisualizer.visualize_filters(self, save_path)


class CNNTrainer:
    def __init__(self, model, initial_lr=1e-3, batch_size=64, lr_schedule=None):
        self.model = model
        self.initial_lr = initial_lr
        self.batch_size = batch_size
        self.visualization_dir = "weight_visualizations"

        if not os.path.exists(self.visualization_dir):
            os.makedirs(self.visualization_dir)

        # Default learning rate schedule if none provided
        self.lr_schedule = lr_schedule or [
            (50, 1.0),  # Initial learning rate
            (30, 0.1),  # 10% of initial lr
            (20, 0.01)  # 1% of initial lr
        ]

        self.augmentation = Compose([
            HorizontalFlip(probability=0.5),
            RandomRotation(angle_range=(-15, 15), probability=0.2),
        ])

        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.test_losses = []
        self.test_accuracies = []

    def compute_accuracy(self, outputs, targets):
        predictions = np.argmax(outputs, axis=0)
        true_labels = np.argmax(targets, axis=0)
        return np.mean(predictions == true_labels)

    def train(self, x_train, y_train, x_val, y_val, x_test, y_test):
        # Normalize data
        mean = np.mean(x_train, axis=(0, 2, 3), keepdims=True)
        std = np.std(x_train, axis=(0, 2, 3), keepdims=True) + 1e-5

        x_train = (x_train - mean) / std
        x_val = (x_val - mean) / std
        x_test = (x_test - mean) / std

        m = x_train.shape[0]
        num_batches = m // self.batch_size
        current_epoch = 0
        total_epochs = sum(epochs for epochs, _ in self.lr_schedule)

        for epochs, factor in self.lr_schedule:
            lr = self.initial_lr * factor
            print(f"\nStarting training with learning rate: {lr}")

            for epoch in range(epochs):
                # Shuffle training data
                indices = np.random.permutation(m)
                x_train_shuffled = x_train[indices]
                y_train_shuffled = y_train[:, indices]

                epoch_loss = 0
                epoch_accuracy = 0

                print(f"\nEpoch {current_epoch + 1}/{total_epochs}")
                print("=" * 50)

                for j in range(num_batches):
                    start_idx = j * self.batch_size
                    end_idx = start_idx + self.batch_size

                    batch_x = x_train_shuffled[start_idx:end_idx]
                    batch_y = y_train_shuffled[:, start_idx:end_idx]

                    augmented_batch_x = self.augmentation(batch_x)

                    # Forward pass
                    outputs = self.model.forward(augmented_batch_x, training=True)
                    loss = self.model.compute_loss(outputs, batch_y)
                    accuracy = self.compute_accuracy(outputs, batch_y)

                    # Backward pass and update
                    gradients = self.model.backward(augmented_batch_x, batch_y)
                    self.model.update_parameters(gradients, lr)

                    # Record batch metrics
                    epoch_loss += loss
                    epoch_accuracy += accuracy

                    # Print progress every 10 batches
                    if (j + 1) % 10 == 0:
                        print(f"Batch {j+1}/{num_batches}")
                        print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")



                # Calculate epoch metrics
                epoch_loss /= num_batches
                epoch_accuracy /= num_batches

                current_epoch += 1

                # Validation metrics
                val_outputs = self.model.forward(x_val, training=False)
                val_loss = self.model.compute_loss(val_outputs, y_val)
                val_accuracy = self.compute_accuracy(val_outputs, y_val)

                # Test metrics
                test_outputs = self.model.forward(x_test, training=False)
                test_loss = self.model.compute_loss(test_outputs, y_test)
                test_accuracy = self.compute_accuracy(test_outputs, y_test)

                # Record metrics
                self.train_losses.append(epoch_loss)
                self.train_accuracies.append(epoch_accuracy)
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_accuracy)
                self.test_losses.append(test_loss)
                self.test_accuracies.append(test_accuracy)

                print("\nEpoch Summary:")
                print(f"Train Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
                print(f"Val Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
                print(f"Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")

        return (self.train_losses, self.train_accuracies,
                self.val_losses, self.val_accuracies,
                self.test_losses, self.test_accuracies)



# Example usage
if __name__ == "__main__":

    x_train, y_train, x_val, y_val, x_test, y_test = DataLoader.load_cifar10_data(total_images=10000)

    model = CNN()

    # Create trainer with custom learning rate schedule
    trainer = CNNTrainer(
        model=model,
        initial_lr=1e-3,
        batch_size=32,
        lr_schedule=[
            (1, 1.0),
            (1, 0.8),# Initial learning rate for 50 epochs
            (1, (0.1)*4),  # 10% of initial lr for 30 epochs
            (1, 0.01)  # 1% of initial lr for 20 epochs
        ]
    )

    # Train model
    results = trainer.train(x_train, y_train, x_val, y_val, x_test, y_test)

    # Visualize results
    Visualizer.plot_results(*results)