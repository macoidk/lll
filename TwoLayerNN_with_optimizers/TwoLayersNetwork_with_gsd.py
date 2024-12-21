#0.50

import numpy as np
from Data_loader import DataLoader
from Activation_Functions import ActivationFunctions
from Visualizer import Visualizer
from LossFunctions import CrossEntropyLoss, L2RegularizationLoss
from Optimizers import SGD


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, num_classes,
                 activation=ActivationFunctions.leaky_relu,
                 momentum=0.9,
                 lambda_reg=0.01):
        # Initialize network parameters
        self.parameters = {
            'W1': np.random.randn(hidden_size, input_size) / np.sqrt(input_size),
            'b1': np.zeros((hidden_size, 1)),
            'W2': np.random.randn(num_classes, hidden_size) / np.sqrt(hidden_size),
            'b2': np.zeros((num_classes, 1))
        }

        # Set activation function
        self.activation = activation
        self.activation_backward = {
            ActivationFunctions.leaky_relu: ActivationFunctions.leaky_relu_backward,
            ActivationFunctions.relu: ActivationFunctions.relu_backward
        }[activation]

        # Initialize loss functions
        self.cross_entropy = CrossEntropyLoss()
        self.l2_reg = L2RegularizationLoss()
        self.lambda_reg = lambda_reg

        # Initialize optimizer
        self.optimizer = SGD(momentum=momentum)
        self.optimizer.initialize(self.parameters)

    def forward_propagation(self, X):
        """
        Forward pass through the network
        """
        # First layer
        Z1 = np.dot(self.parameters['W1'], X) + self.parameters['b1']
        A1 = self.activation(Z1)

        # Second layer
        Z2 = np.dot(self.parameters['W2'], A1) + self.parameters['b2']
        A2 = ActivationFunctions.softmax(Z2)

        cache = (Z1, A1, Z2, A2)
        return A2, cache

    def backward_propagation(self, X, Y, cache):
        """
        Backward pass to compute gradients
        """
        m = X.shape[1]
        (Z1, A1, Z2, A2) = cache

        # Output layer gradients
        dZ2 = self.cross_entropy.backward(A2, Y)
        dW2 = 1 / m * np.dot(dZ2, A1.T)
        db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)

        # Add L2 regularization gradient
        dW2 += self.l2_reg.backward(self.parameters['W2'], self.lambda_reg)

        # Hidden layer gradients
        dA1 = np.dot(self.parameters['W2'].T, dZ2)
        dZ1 = self.activation_backward(dA1, Z1)
        dW1 = 1 / m * np.dot(dZ1, X.T)
        db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

        # Add L2 regularization gradient
        dW1 += self.l2_reg.backward(self.parameters['W1'], self.lambda_reg)

        gradients = {
            'W1': dW1, 'b1': db1,
            'W2': dW2, 'b2': db2
        }

        return gradients

    def update_parameters(self, gradients, learning_rate):
        """Update parameters using the optimizer"""
        self.optimizer.update(gradients, learning_rate)


class ModelTrainer:
    def __init__(self, model, X_train, Y_train, X_test, Y_test,
                 initial_lr=1e-3,
                 batch_size=64,
                 lr_schedule=None):
        self.model = model
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.initial_lr = initial_lr
        self.batch_size = batch_size

        # Default learning rate schedule
        self.lr_schedule = lr_schedule or [
            (120, 1.0),  # Initial learning rate for 120 epochs
            (50, 0.04),  # 4% of initial lr for 50 epochs
            (50, 0.001),  # 0.1% of initial lr for 50 epochs
            (50, 0.0001)  # 0.01% of initial lr for 50 epochs
        ]

        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []

    def compute_loss(self, A2, Y):
        """Compute total loss including regularization"""
        ce_loss = self.model.cross_entropy.forward(A2, Y)
        l2_loss = self.model.l2_reg.forward(self.model.parameters, self.model.lambda_reg)
        return ce_loss + l2_loss

    def compute_accuracy(self, X, Y):
        """Compute classification accuracy"""
        A2, _ = self.model.forward_propagation(X)
        predictions = np.argmax(A2, axis=0)
        true_labels = np.argmax(Y, axis=0)
        return np.mean(predictions == true_labels)

    def train_model_with_lr_decay(self):
        """Train the model using learning rate decay schedule"""
        m = self.X_train.shape[1]
        num_batches = m // self.batch_size

        if isinstance(self.lr_schedule[0], (int, float)):
            epochs = self.lr_schedule
            factors = [0.1 ** (i + 1) for i in range(len(epochs))]
            self.lr_schedule = list(zip(epochs, factors))

        current_epoch = 0

        for epochs, factor in self.lr_schedule:
            lr = self.initial_lr * factor
            for _ in range(epochs):
                # Shuffle training data
                permutation = np.random.permutation(m)
                X_shuffled = self.X_train[:, permutation]
                Y_shuffled = self.Y_train[:, permutation]

                for j in range(num_batches):
                    start = j * self.batch_size
                    end = start + self.batch_size
                    X_batch = X_shuffled[:, start:end]
                    Y_batch = Y_shuffled[:, start:end]

                    A2, cache = self.model.forward_propagation(X_batch)
                    gradients = self.model.backward_propagation(X_batch, Y_batch, cache)
                    self.model.update_parameters(gradients, lr)

                current_epoch += 1
                if current_epoch % 10 == 0:
                    self.record_metrics()
                    self.print_progress(current_epoch, lr)

        return (self.train_losses, self.train_accuracies,
                self.test_losses, self.test_accuracies)

    def record_metrics(self):
        """Record training and testing metrics"""
        train_A2, _ = self.model.forward_propagation(self.X_train)
        test_A2, _ = self.model.forward_propagation(self.X_test)

        train_loss = self.compute_loss(train_A2, self.Y_train)
        test_loss = self.compute_loss(test_A2, self.Y_test)

        train_accuracy = self.compute_accuracy(self.X_train, self.Y_train)
        test_accuracy = self.compute_accuracy(self.X_test, self.Y_test)

        self.train_losses.append(train_loss)
        self.test_losses.append(test_loss)
        self.train_accuracies.append(train_accuracy)
        self.test_accuracies.append(test_accuracy)

    def print_progress(self, epoch, lr):
        """Print training progress"""
        print(f"Epoch {epoch}, LR: {lr:.1e}, "
              f"Train Loss: {self.train_losses[-1]:.4f}, "
              f"Train Accuracy: {self.train_accuracies[-1]:.4f}, "
              f"Test Loss: {self.test_losses[-1]:.4f}, "
              f"Test Accuracy: {self.test_accuracies[-1]:.4f}")


if __name__ == "__main__":
    print("Starting the program...")
    print("Loading data...")
    x_train, y_train, x_test, y_test = DataLoader.load_cifar10_data()
    x_train = x_train[:, :37000]
    y_train = y_train[:, :37000]
    print("Data loaded successfully.")

    # Network configuration
    input_size = x_train.shape[0]
    hidden_size = 64
    num_classes = y_train.shape[0]

    # Create and train model
    model = NeuralNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        num_classes=num_classes,
        momentum=0.9,
        lambda_reg=0.01
    )

    trainer = ModelTrainer(
        model=model,
        X_train=x_train,
        Y_train=y_train,
        X_test=x_test,
        Y_test=y_test,
        initial_lr=1e-3,
        batch_size=64
    )

    print("Starting training...")
    results = trainer.train_model_with_lr_decay()
    print("Training completed.")

    # Plot results
    Visualizer.plot_results(*results)

    print(f"Final train accuracy: {trainer.train_accuracies[-1]:.4f}")
    print(f"Final test accuracy: {trainer.test_accuracies[-1]:.4f}")