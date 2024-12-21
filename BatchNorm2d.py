#BatchNorm2d
import numpy as np


class BatchNorm2d:
    def __init__(self, num_features, epsilon=1e-5, momentum=0.9):
        self.num_features = num_features
        self.epsilon = epsilon
        self.momentum = momentum
        self.gamma = np.ones((1, num_features, 1, 1))
        self.beta = np.zeros((1, num_features, 1, 1))
        self.running_mean = np.zeros((1, num_features, 1, 1))
        self.running_var = np.ones((1, num_features, 1, 1))
        self.cache = None
        self.training = True

    def forward(self, x, training=True):
        self.training = training
        N, C, H, W = x.shape

        if training:
            # Vectorized mean and variance calculation
            batch_mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
            batch_var = np.var(x, axis=(0, 2, 3), keepdims=True)

            # Vectorized normalization
            x_normalized = (x - batch_mean) / np.sqrt(batch_var + self.epsilon)
            out = self.gamma * x_normalized + self.beta

            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var

            self.cache = {
                'x': x,
                'normalized': x_normalized,
                'mean': batch_mean,
                'var': batch_var,
                'sqrt_var': np.sqrt(batch_var + self.epsilon)
            }
        else:
            x_normalized = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            out = self.gamma * x_normalized + self.beta

        return out

    def backward(self, dout):
        if not self.training:
            raise RuntimeError("Backward pass called during inference")

        x = self.cache['x']
        normalized = self.cache['normalized']
        mean = self.cache['mean']
        var = self.cache['var']
        sqrt_var = self.cache['sqrt_var']
        N, C, H, W = x.shape
        M = N * H * W

        # Vectorized gradient calculations
        dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)
        dgamma = np.sum(dout * normalized, axis=(0, 2, 3), keepdims=True)

        dx_normalized = dout * self.gamma
        dvar = np.sum(dx_normalized * (x - mean) * -0.5 * (var + self.epsilon) ** (-1.5),
                      axis=(0, 2, 3), keepdims=True)

        dmean = np.sum(dx_normalized * -1 / sqrt_var, axis=(0, 2, 3), keepdims=True) + \
                dvar * np.mean(-2 * (x - mean), axis=(0, 2, 3), keepdims=True)

        dx = dx_normalized / sqrt_var + \
             dvar * 2 * (x - mean) / M + \
             dmean / M

        self.cache.update({'dgamma': dgamma, 'dbeta': dbeta})
        return dx, dgamma, dbeta

    def get_parameters(self):
        """
        Get trainable parameters

        Returns:
            dict: Dictionary containing gamma and beta parameters
        """
        return {
            'gamma': self.gamma,
            'beta': self.beta
        }

    def set_parameters(self, params):
        """
        Set trainable parameters

        Args:
            params (dict): Dictionary containing gamma and beta parameters
        """
        self.gamma = params['gamma']
        self.beta = params['beta']