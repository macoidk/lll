import numpy as np
from SNNData_loader import DataLoader
from Activation_Functions import ActivationFunctions
from Visualizer import Visualizer
from Optimizers import Adam
from BatchNormalization import BatchNormalization

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, num_classes,
                 activation=ActivationFunctions.leaky_relu,
                 beta1=0.9, beta2=0.999,
                 lambda_reg=0.5):
        # Initialize network parameters
        self.parameters = {
            'W1': np.random.randn(hidden_size, input_size) / np.sqrt(input_size),
            'b1': np.zeros((hidden_size, 1)),
            'W2': np.random.randn(num_classes, hidden_size) / np.sqrt(hidden_size),
            'b2': np.zeros((num_classes, 1))
        }

        self.bn1 = BatchNormalization(num_features=hidden_size)


        # Set activation function
        self.activation = activation
        self.activation_backward = {
            ActivationFunctions.leaky_relu: ActivationFunctions.leaky_relu_backward,
            ActivationFunctions.relu: ActivationFunctions.relu_backward
        }[activation]

        # Initialize optimizer
        self.optimizer = Adam(beta1=beta1, beta2=beta2)

        self.parameters.update({
            'gamma1': self.bn1.gamma,
            'beta1': self.bn1.beta
        })

        self.optimizer.initialize(self.parameters)

        # Regularization strength
        self.lambda_reg = lambda_reg

    def forward_propagation(self, X):
        """Forward pass through the network"""
        Z1 = np.dot(self.parameters['W1'], X) + self.parameters['b1']

        Z1_bn = self.bn1.forward(Z1)

        A1 = self.activation(Z1_bn)
        Z2 = np.dot(self.parameters['W2'], A1) + self.parameters['b2']
        A2 = ActivationFunctions.softmax(Z2)

        cache = (Z1, Z1_bn, A1, Z2, A2)
        return A2, cache

    def compute_loss(self, A2, Y, m):
        """Compute cross-entropy loss with L2 regularization"""
        # Cross-entropy loss
        epsilon = 1e-8
        ce_loss = -1 / m * np.sum(Y * np.log(A2 + epsilon))

        # L2 regularization
        l2_loss = self.lambda_reg / (2 * m) * (
                np.sum(np.square(self.parameters['W1'])) +
                np.sum(np.square(self.parameters['W2']))
        )

        return ce_loss + l2_loss

    def backward_propagation(self, X, Y, cache):
        """Backward pass to compute gradients"""
        m = X.shape[1]
        Z1, Z1_bn, A1, Z2, A2 = cache

        # Output layer gradients
        dZ2 = A2 - Y
        dW2 = 1 / m * np.dot(dZ2, A1.T) + (self.lambda_reg / m) * self.parameters['W2']
        db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)

        # Hidden layer gradients
        dA1 = np.dot(self.parameters['W2'].T, dZ2)

        dZ1_bn = self.activation_backward(dA1, Z1_bn)

        dZ1, dgamma1, dbeta1 = self.bn1.backward(dZ1_bn)

        dW1 = 1 / m * np.dot(dZ1, X.T) + (self.lambda_reg / m) * self.parameters['W1']
        db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

        return {
            'W1': dW1, 'b1': db1,
            'W2': dW2, 'b2': db2,
            'gamma1': dgamma1, 'beta1': dbeta1
        }

    def update_parameters(self, gradients, learning_rate):
        """Update parameters using Adam optimizer"""
        self.optimizer.update(gradients, learning_rate)
        self.bn1.gamma = self.parameters['gamma1']
        self.bn1.beta = self.parameters['beta1']


class ModelTrainer:
    def __init__(self, model, X_train, Y_train, X_test, Y_test, X_val, Y_val,
                 initial_lr=1e-3,
                 batch_size=64,
                 lr_schedule=None):
        self.model = model
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.X_val = X_val
        self.Y_val = Y_val
        self.initial_lr = initial_lr
        self.batch_size = batch_size

        # Default learning rate schedule if none provided
        self.lr_schedule = lr_schedule or [
            (50, 1.0),  # Initial learning rate for 120 epochs
            (30, 0.1),  # 10% of initial lr for 50 epochs
            (20, 0.01),  # 1% of initial lr for 50 epochs
            (10, 0.001)  # 0.1% of initial lr for 50 epochs
        ]

        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

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
        current_epoch = 0

        for epochs, factor in self.lr_schedule:
            lr = self.initial_lr * factor

            for _ in range(epochs):
                # Shuffle training data
                permutation = np.random.permutation(m)
                X_shuffled = self.X_train[:, permutation]
                Y_shuffled = self.Y_train[:, permutation]

                for j in range(num_batches):
                    # Get mini-batch
                    start = j * self.batch_size
                    end = start + self.batch_size
                    X_batch = X_shuffled[:, start:end]
                    Y_batch = Y_shuffled[:, start:end]

                    # Forward propagation
                    A2, cache = self.model.forward_propagation(X_batch)

                    # Backward propagation
                    gradients = self.model.backward_propagation(X_batch, Y_batch, cache)

                    # Update parameters
                    self.model.update_parameters(gradients, lr)

                current_epoch += 1
                if current_epoch % 10 == 0:
                    self.record_metrics()
                    self.print_progress(current_epoch, lr)

        return (self.train_losses, self.train_accuracies,
                self.val_losses, self.val_accuracies,
                self.test_losses, self.test_accuracies)

    def record_metrics(self):
        """Record training and testing metrics"""
        # Training metrics
        train_A2, train_cache = self.model.forward_propagation(self.X_train)
        train_loss = self.model.compute_loss(train_A2, self.Y_train, self.X_train.shape[1])
        train_accuracy = self.compute_accuracy(self.X_train, self.Y_train)

        val_A2, val_cache = self.model.forward_propagation(self.X_val)
        val_loss = self.model.compute_loss(val_A2, self.Y_val, self.X_val.shape[1])
        val_accuracy = self.compute_accuracy(self.X_val, self.Y_val)

        # Testing metrics
        test_A2, test_cache = self.model.forward_propagation(self.X_test)
        test_loss = self.model.compute_loss(test_A2, self.Y_test, self.X_test.shape[1])
        test_accuracy = self.compute_accuracy(self.X_test, self.Y_test)

        # Record metrics
        self.train_losses.append(train_loss)
        self.test_losses.append(test_loss)
        self.val_losses.append(val_loss)
        self.train_accuracies.append(train_accuracy)
        self.test_accuracies.append(test_accuracy)
        self.val_accuracies.append(val_accuracy)


    def print_progress(self, epoch, lr):
        """Print training progress"""
        print(f"Epoch {epoch}, LR: {lr:.1e}, "
              f"Train Loss: {self.train_losses[-1]:.4f}, "
              f"Train Accuracy: {self.train_accuracies[-1]:.4f}, "
              f"Val Loss: {self.val_losses[-1]:.4f}, "
              f"Val Accuracy: {self.val_accuracies[-1]:.4f}, "
              f"Test Loss: {self.test_losses[-1]:.4f}, "
              f"Test Accuracy: {self.test_accuracies[-1]:.4f}")




if __name__ == "__main__":
    print("Starting the program...")
    print("Loading data...")
    x_train, y_train, x_val, y_val, x_test, y_test = DataLoader.load_cifar10_data()
    print("Data loaded successfully.")

    # Network configuration
    input_size = x_train.shape[0]
    hidden_size = 256
    num_classes = y_train.shape[0]

    # Create model with Adam optimizer
    model = NeuralNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        num_classes=num_classes,
        beta1=0.9,
        beta2=0.999,
        lambda_reg=0.01
    )

    custom_lr_schedule = [
        (50, 1.0),  # Initial learning rate for 60 epochs
        (30, 0.1),  # 10% of initial lr for 40 epochs
        (20, 0.01),  # 1% of initial lr for 20 epochs
    ]

    # Create trainer
    trainer = ModelTrainer(
        model=model,
        X_train=x_train,
        Y_train=y_train,
        X_val=x_val,
        Y_val=y_val,
        X_test=x_test,
        Y_test=y_test,
        initial_lr=1e-3,
        batch_size=500,
        lr_schedule=custom_lr_schedule
    )

    print("Starting training...")
    results = trainer.train_model_with_lr_decay()
    print("Training completed.")

    # Plot results
    Visualizer.plot_results(*results)

    print(f"Final train accuracy: {trainer.train_accuracies[-1]:.4f}")
    print(f"Final test accuracy: {trainer.test_accuracies[-1]:.4f}")
    print(f"Final val accuracy: {trainer.val_accuracies[-1]:.4f}")
