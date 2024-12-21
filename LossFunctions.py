#LossFunction
import numpy as np


class LossFunction:
    @staticmethod
    def forward(y_pred, y_true):
        raise NotImplementedError

    @staticmethod
    def backward(y_pred, y_true):
        raise NotImplementedError


class CrossEntropyLoss(LossFunction):
    @staticmethod
    def forward(y_pred, y_true, epsilon=1e-15):
        """
        Computes the cross-entropy loss between predictions and true labels

        Parameters:
        y_pred: predicted probabilities (after softmax) - shape (n_classes, batch_size)
        y_true: true labels (one-hot encoded) - shape (n_classes, batch_size)
        epsilon: small constant to avoid log(0)

        Returns:
        float: mean cross-entropy loss
        """
        # Clip predictions to avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=0))

    @staticmethod
    def backward(y_pred, y_true):
        """
        Computes gradient of cross-entropy loss with respect to predictions

        Returns:
        array: gradient of the same shape as y_pred
        """
        return y_pred - y_true


class L2RegularizationLoss(LossFunction):
    @staticmethod
    def forward(weights, lambda_reg):
        """
        Computes L2 regularization loss

        Parameters:
        weights: dictionary of weight matrices
        lambda_reg: regularization strength

        Returns:
        float: L2 regularization loss
        """
        reg_loss = 0
        for w in weights.values():
            reg_loss += np.sum(np.square(w))
        return 0.5 * lambda_reg * reg_loss

    @staticmethod
    def backward(weight, lambda_reg):
        """
        Computes gradient of L2 regularization with respect to weights

        Returns:
        array: gradient of the same shape as weight
        """
        return lambda_reg * weight


class MSELoss(LossFunction):
    @staticmethod
    def forward(y_pred, y_true):
        """
        Computes Mean Squared Error loss

        Parameters:
        y_pred: predicted values - shape (n_outputs, batch_size)
        y_true: true values - shape (n_outputs, batch_size)

        Returns:
        float: mean squared error
        """
        return np.mean(np.sum(np.square(y_pred - y_true), axis=0))

    @staticmethod
    def backward(y_pred, y_true):
        """
        Computes gradient of MSE with respect to predictions

        Returns:
        array: gradient of the same shape as y_pred
        """
        return 2 * (y_pred - y_true) / y_pred.shape[1]


class BinaryCrossEntropyLoss(LossFunction):
    @staticmethod
    def forward(y_pred, y_true, epsilon=1e-15):
        """
        Computes Binary Cross-Entropy loss

        Parameters:
        y_pred: predicted probabilities - shape (1, batch_size)
        y_true: true labels (0 or 1) - shape (1, batch_size)
        epsilon: small constant to avoid log(0)

        Returns:
        float: mean binary cross-entropy loss
        """
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    @staticmethod
    def backward(y_pred, y_true, epsilon=1e-15):
        """
        Computes gradient of binary cross-entropy loss

        Returns:
        array: gradient of the same shape as y_pred
        """
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / y_pred.shape[1]


class HingeLoss(LossFunction):
    @staticmethod
    def forward(y_pred, y_true):
        """
        Computes Hinge loss (for SVM)

        Parameters:
        y_pred: predicted scores - shape (n_classes, batch_size)
        y_true: true labels (one-hot encoded) - shape (n_classes, batch_size)

        Returns:
        float: mean hinge loss
        """
        margins = np.maximum(0, 1 - y_true * y_pred)
        return np.mean(np.sum(margins, axis=0))

    @staticmethod
    def backward(y_pred, y_true):
        """
        Computes gradient of hinge loss

        Returns:
        array: gradient of the same shape as y_pred
        """
        margins = 1 - y_true * y_pred
        grad = np.zeros_like(y_pred)
        grad[margins > 0] = -y_true[margins > 0]
        return grad / y_pred.shape[1]


class HuberLoss(LossFunction):
    @staticmethod
    def forward(y_pred, y_true, delta=1.0):
        """
        Computes Huber loss (smooth L1 loss)

        Parameters:
        y_pred: predicted values - shape (n_outputs, batch_size)
        y_true: true values - shape (n_outputs, batch_size)
        delta: threshold parameter

        Returns:
        float: mean huber loss
        """
        error = y_pred - y_true
        is_small_error = np.abs(error) <= delta

        squared_loss = 0.5 * np.square(error)
        linear_loss = delta * np.abs(error) - 0.5 * delta ** 2

        return np.mean(np.sum(
            np.where(is_small_error, squared_loss, linear_loss),
            axis=0
        ))

    @staticmethod
    def backward(y_pred, y_true, delta=1.0):
        """
        Computes gradient of Huber loss

        Returns:
        array: gradient of the same shape as y_pred
        """
        error = y_pred - y_true
        is_small_error = np.abs(error) <= delta

        grad = np.where(is_small_error,
                        error,
                        delta * np.sign(error))
        return grad / y_pred.shape[1]