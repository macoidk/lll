import numpy as np


class ActivationFunctions:
    @staticmethod
    def leaky_relu(Z, alpha=0.01):
        return np.maximum(alpha * Z, Z)

    @staticmethod
    def relu(Z):
        return np.maximum(0, Z)

    @staticmethod
    def leaky_relu_backward(dA, Z, alpha=0.01):
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] *= alpha
        return dZ

    @staticmethod
    def relu_backward(dA, Z):
        """
        Compute gradient of ReLU activation function

        Parameters:
        dA: upstream derivatives
        Z: input to ReLU function during forward pass

        Returns:
        dZ: gradient of ReLU with respect to Z
        """
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ

    @staticmethod
    def softmax(Z):
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)