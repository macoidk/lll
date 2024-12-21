#Optimizers.py
import numpy as np


class Optimizer:
    def __init__(self):
        self.parameters = None

    def initialize(self, parameters):
        self.parameters = parameters


class SGD(Optimizer):
    def __init__(self, momentum=0.9):
        super().__init__()
        self.momentum = momentum
        self.v = None

    def initialize(self, parameters):
        super().initialize(parameters)
        self.v = {key: np.zeros_like(value) for key, value in parameters.items()}

    def update(self, gradients, learning_rate):
        for key in self.parameters:
            self.v[key] = self.momentum * self.v[key] + learning_rate * gradients[key]
            self.parameters[key] -= self.v[key]


class Adam(Optimizer):
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__()
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def initialize(self, parameters):
        super().initialize(parameters)
        self.m = {key: np.zeros_like(value) for key, value in parameters.items()}
        self.v = {key: np.zeros_like(value) for key, value in parameters.items()}

    def update(self, gradients, learning_rate):
        self.t += 1
        for key in self.parameters:
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * gradients[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * np.square(gradients[key])

            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            self.parameters[key] -= learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)


class RMSprop(Optimizer):
    def __init__(self, decay_rate=0.9, epsilon=1e-8):
        super().__init__()
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache = None

    def initialize(self, parameters):
        super().initialize(parameters)
        self.cache = {key: np.zeros_like(value) for key, value in parameters.items()}

    def update(self, gradients, learning_rate):
        for key in self.parameters:
            self.cache[key] = self.decay_rate * self.cache[key] + (1 - self.decay_rate) * np.square(gradients[key])
            self.parameters[key] -= learning_rate * gradients[key] / (np.sqrt(self.cache[key]) + self.epsilon)


class Nesterov(Optimizer):
    def __init__(self, momentum=0.9):
        super().__init__()
        self.momentum = momentum
        self.v = None

    def initialize(self, parameters):
        super().initialize(parameters)
        self.v = {key: np.zeros_like(value) for key, value in parameters.items()}

    def update(self, gradients, learning_rate):
        for key in self.parameters:
            v_prev = self.v[key]
            self.v[key] = self.momentum * self.v[key] - learning_rate * gradients[key]
            self.parameters[key] += -self.momentum * v_prev + (1 + self.momentum) * self.v[key]