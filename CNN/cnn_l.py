#model 1
#conv 3
#accuracy = 0.57
#time to train = 30m
#filters initialization - He
#FC1 512
#FC2 10

import numpy as np
from Layers import Conv2d, Linear, Flatten, MaxPool2d
from BatchNorm2d import BatchNorm2d
from LossFunctions import CrossEntropyLoss
from Optimizers import Adam
from CNNData_loader import DataLoader
from Visualizer import Visualizer, FilterVisualizer
from LossFunctions import L2RegularizationLoss


class CNN:
    def __init__(self):
        # Network architecture
        self.conv1 = Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = BatchNorm2d(32)
        self.pool1 = MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = BatchNorm2d(64)
        self.pool2 = MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = BatchNorm2d(64)
        self.pool3 = MaxPool2d(kernel_size=2, stride=2)

        self.flatten = Flatten()
        self.fc1 = Linear(in_features=64 * 4 * 4, out_features=256)
        self.fc2 = Linear(in_features=256, out_features=10)

        # Loss functions and optimizer
        self.criterion = CrossEntropyLoss()
        self.l2_reg = L2RegularizationLoss()
        self.optimizer = Adam(beta1=0.9, beta2=0.999)
        self.learning_rate = 0.001  # Reduced learning rate
        self.lambda_reg = 0.0001  # L2 regularization strength 0.0001

        # Initialize optimizer
        self.optimizer.initialize({
            'conv1_w': self.conv1.W, 'conv1_b': self.conv1.b,
            'conv2_w': self.conv2.W, 'conv2_b': self.conv2.b,
            'conv3_w': self.conv3.W, 'conv3_b': self.conv3.b,
            'fc1_w': self.fc1.W, 'fc1_b': self.fc1.b,
            'fc2_w': self.fc2.W, 'fc2_b': self.fc2.b,
            'bn1_gamma': self.bn1.gamma, 'bn1_beta': self.bn1.beta,
            'bn2_gamma': self.bn2.gamma, 'bn2_beta': self.bn2.beta,
            'bn3_gamma': self.bn3.gamma, 'bn3_beta': self.bn3.beta
        })

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

        # Fully connected layers
        x = self.flatten.forward(x)
        self.activations['flatten'] = x
        x = x.T
        x = self.fc1.forward(x)
        self.activations['fc1'] = x
        x = np.maximum(0, x)  # ReLU
        self.activations['relu4'] = x

        x = self.fc2.forward(x)
        self.activations['fc2'] = x

        # Softmax with numerical stability
        x = x - np.max(x, axis=0, keepdims=True)  # For numerical stability
        exp_scores = np.exp(x)
        probs = exp_scores / np.sum(exp_scores, axis=0, keepdims=True)
        self.activations['softmax'] = probs
        return probs

    def backward(self, x, y):
        # Compute gradients
        dout = self.criterion.backward(self.activations['softmax'], y)

        # Backward pass through the network
        dout = self.fc2.backward(dout)

        # ReLU backward for fc1
        dout = dout * (self.activations['relu4'] > 0)
        dout = self.fc1.backward(dout)

        # Reshape for convolution layers
        dout = self.flatten.backward(dout)

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
        weights = {
            'conv1_w': self.conv1.W,
            'conv2_w': self.conv2.W,
            'conv3_w': self.conv3.W,
            'fc1_w': self.fc1.W,
            'fc2_w': self.fc2.W
        }

        # Add L2 regularization gradients to weight gradients
        self.conv1.dW += self.l2_reg.backward(self.conv1.W, self.lambda_reg)
        self.conv2.dW += self.l2_reg.backward(self.conv2.W, self.lambda_reg)
        self.conv3.dW += self.l2_reg.backward(self.conv3.W, self.lambda_reg)
        self.fc1.dW += self.l2_reg.backward(self.fc1.W, self.lambda_reg)
        self.fc2.dW += self.l2_reg.backward(self.fc2.W, self.lambda_reg)

        # Update parameters
        self.optimizer.update({
            'conv1_w': self.conv1.dW, 'conv1_b': self.conv1.db,
            'conv2_w': self.conv2.dW, 'conv2_b': self.conv2.db,
            'conv3_w': self.conv3.dW, 'conv3_b': self.conv3.db,
            'fc1_w': self.fc1.dW, 'fc1_b': self.fc1.db,
            'fc2_w': self.fc2.dW, 'fc2_b': self.fc2.db,
            'bn1_gamma': dgamma1, 'bn1_beta': dbeta1,
            'bn2_gamma': self.bn2.cache['dgamma'], 'bn2_beta': self.bn2.cache['dbeta'],
            'bn3_gamma': self.bn3.cache['dgamma'], 'bn3_beta': self.bn3.cache['dbeta']
        }, learning_rate=self.learning_rate)

        # Return regularization loss
        return self.l2_reg.forward(weights, self.lambda_reg)

    def train(self, num_epochs=10, batch_size=32):
        # Load and preprocess CIFAR-10 data
        x_train, y_train, x_val, y_val, x_test, y_test = DataLoader.load_cifar10_data()

        # Normalize data
        mean = np.mean(x_train, axis=(0, 2, 3), keepdims=True)
        std = np.std(x_train, axis=(0, 2, 3), keepdims=True) + 1e-5
        x_train = (x_train - mean) / std
        x_val = (x_val - mean) / std
        x_test = (x_test - mean) / std

        train_losses, train_accuracies = [], []
        val_losses, val_accuracies = [], []
        test_losses, test_accuracies = [], []

        num_samples = len(x_train)
        num_batches = num_samples // batch_size

        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_accuracy = 0

            # Shuffle training data
            indices = np.random.permutation(num_samples)
            x_train = x_train[indices]
            y_train_epoch = y_train[:, indices]

            # Training
            for batch in range(num_batches):
                start_idx = batch * batch_size
                end_idx = start_idx + batch_size

                batch_x = x_train[start_idx:end_idx]
                batch_y = y_train_epoch[:, start_idx:end_idx]

                # Forward and backward pass
                output = self.forward(batch_x)
                loss = self.criterion.forward(output, batch_y)
                reg_loss = self.backward(batch_x, batch_y)
                total_loss = loss + reg_loss

                # Calculate accuracy
                predictions = np.argmax(output, axis=0)
                true_labels = np.argmax(batch_y, axis=0)
                accuracy = np.mean(predictions == true_labels)

                epoch_loss += total_loss
                epoch_accuracy += accuracy

                if batch % 10 == 0:
                    print(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch + 1}/{num_batches}")
                    print(f"Training Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")

            # Calculate epoch averages
            epoch_loss /= num_batches
            epoch_accuracy /= num_batches
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_accuracy)

            # Validation
            val_output = self.forward(x_val, training=False)
            val_loss = self.criterion.forward(val_output, y_val)
            val_pred = np.argmax(val_output, axis=0)
            val_true = np.argmax(y_val, axis=0)
            val_accuracy = np.mean(val_pred == val_true)

            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"Training Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
            print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}\n")



        # Visualize results
        Visualizer.plot_results(
            train_losses, train_accuracies,
            val_losses, val_accuracies,
            test_losses, test_accuracies
        )

    def visualize_network_filters(self, save_path=None):
        """
        Visualize filters from all convolutional layers of the network

        Args:
            save_path (str, optional): Path to save the visualization plot
        """
        FilterVisualizer.visualize_filters(self, save_path)


# Example usage
if __name__ == "__main__":
    model = CNN()
    model.train(num_epochs=1, batch_size=64)
    model.visualize_network_filters(save_path="cnn_filters.png")
