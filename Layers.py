#Layers.py
import numpy as np
from typing import Tuple, Optional

class Layer:
    """Base class for all neural network layers"""

    def __init__(self):
        self.training = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, dout: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_parameters(self) -> dict:
        return {}

    def set_parameters(self, params: dict) -> None:
        pass

    def train(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False


class Linear(Layer):
    """Fully connected layer implementation"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        Initialize Linear layer

        Args:
            in_features: Number of input features
            out_features: Number of output features
            bias: Whether to include bias
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        # Initialize weights using He initialization
        self.W = np.random.randn(out_features, in_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros((out_features, 1)) if bias else None

        # Cache for backward pass
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass

        Args:
            x: Input of shape (in_features, batch_size)

        Returns:
            Output of shape (out_features, batch_size)
        """
        self.x = x
        out = np.dot(self.W, x)
        if self.use_bias:
            out += self.b
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backward pass

        Args:
            dout: Upstream derivatives

        Returns:
            Gradient with respect to input x
        """
        self.dW = np.dot(dout, self.x.T)
        if self.use_bias:
            self.db = np.sum(dout, axis=1, keepdims=True)
        dx = np.dot(self.W.T, dout)
        return dx

    def get_parameters(self) -> dict:
        """Get trainable parameters"""
        params = {'W': self.W}
        if self.use_bias:
            params['b'] = self.b
        return params

    def set_parameters(self, params: dict) -> None:
        """Set trainable parameters"""
        self.W = params['W']
        if self.use_bias:
            self.b = params['b']


class Dropout(Layer):
    """Dropout layer implementation"""

    def __init__(self, p: float = 0.5):
        """
        Initialize Dropout layer

        Args:
            p: Dropout probability
        """
        super().__init__()
        self.p = p
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass

        Args:
            x: Input tensor

        Returns:
            Output with dropped units
        """
        if self.training:
            self.mask = (np.random.rand(*x.shape) > self.p) / (1 - self.p)
            return x * self.mask
        return x

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backward pass

        Args:
            dout: Upstream derivatives

        Returns:
            Gradient with respect to input x
        """
        return dout * self.mask


class Flatten(Layer):
    """Flatten layer implementation"""

    def __init__(self):
        super().__init__()
        self.input_shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass

        Args:
            x: Input tensor of shape (N, C, H, W)

        Returns:
            Flattened output of shape (N, C*H*W)
        """
        self.input_shape = x.shape
        N = x.shape[0]
        return x.reshape(N, -1)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backward pass

        Args:
            dout: Upstream derivatives

        Returns:
            Gradient reshaped to match input shape
        """
        return dout.reshape(self.input_shape)


class Conv2d(Layer):
    """2D Convolutional layer implementation"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0):
        """
        Initialize Conv2d layer

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the convolutional kernel (assumed square)
            stride: Stride of the convolution
            padding: Zero-padding size
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize filters/kernels using He initialization
        self.W = np.random.randn(
            out_channels,
            in_channels,
            kernel_size,
            kernel_size
        ) * np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))

        self.b = np.zeros((out_channels, 1))

        # Cache for backward pass
        self.x = None
        self.x_cols = None

    def _im2col(self, x: np.ndarray) -> np.ndarray:
        """Convert input image to columns for efficient convolution"""
        N, C, H, W = x.shape
        out_h = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (W + 2 * self.padding - self.kernel_size) // self.stride + 1

        # Add padding if needed
        if self.padding > 0:
            x_padded = np.pad(
                x,
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode='constant'
            )
        else:
            x_padded = x

        # Extract patches and reshape
        cols = np.zeros((N, C, self.kernel_size, self.kernel_size, out_h, out_w))
        for y in range(self.kernel_size):
            y_max = y + self.stride * out_h
            for x in range(self.kernel_size):
                x_max = x + self.stride * out_w
                cols[:, :, y, x, :, :] = x_padded[:, :, y:y_max:self.stride, x:x_max:self.stride]

        cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
        return cols

    def _col2im(self, cols: np.ndarray, x_shape: Tuple) -> np.ndarray:
        """Convert columns back to image format"""
        N, C, H, W = x_shape
        H_padded, W_padded = H + 2 * self.padding, W + 2 * self.padding
        out_h = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (W + 2 * self.padding - self.kernel_size) // self.stride + 1

        x_padded = np.zeros((N, C, H_padded, W_padded))
        cols_reshaped = cols.reshape(N, out_h, out_w, C, self.kernel_size, self.kernel_size)
        cols_reshaped = cols_reshaped.transpose(0, 3, 4, 5, 1, 2)

        for y in range(self.kernel_size):
            y_max = y + self.stride * out_h
            for x in range(self.kernel_size):
                x_max = x + self.stride * out_w
                x_padded[:, :, y:y_max:self.stride, x:x_max:self.stride] += \
                    cols_reshaped[:, :, y, x, :, :]

        if self.padding == 0:
            return x_padded
        return x_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of convolution

        Args:
            x: Input of shape (N, C, H, W)

        Returns:
            Output of shape (N, out_channels, H', W')
        """
        self.x = x
        N, C, H, W = x.shape

        # Convert input to columns
        self.x_cols = self._im2col(x)

        # Reshape filters for matrix multiplication
        W_reshaped = self.W.reshape(self.out_channels, -1)

        # Compute convolution as matrix multiplication
        out = np.dot(W_reshaped, self.x_cols.T) + self.b

        # Reshape output
        out_h = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        out = out.reshape(self.out_channels, N, out_h, out_w)
        out = out.transpose(1, 0, 2, 3)

        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backward pass of convolution

        Args:
            dout: Upstream derivatives

        Returns:
            Gradient with respect to input x
        """
        N = self.x.shape[0]

        # Reshape dout
        dout_reshaped = dout.transpose(1, 0, 2, 3).reshape(self.out_channels, -1)

        # Gradient w.r.t. weights
        self.dW = np.dot(dout_reshaped, self.x_cols).reshape(self.W.shape)

        # Gradient w.r.t. bias
        self.db = np.sum(dout_reshaped, axis=1, keepdims=True)

        # Gradient w.r.t. input
        W_reshaped = self.W.reshape(self.out_channels, -1)
        dx_cols = np.dot(W_reshaped.T, dout_reshaped)
        dx = self._col2im(dx_cols.T, self.x.shape)

        return dx

    def get_parameters(self) -> dict:
        """Get trainable parameters"""
        return {
            'W': self.W,
            'b': self.b
        }

    def set_parameters(self, params: dict) -> None:
        """Set trainable parameters"""
        self.W = params['W']
        self.b = params['b']


# Add to Layers.py
class MaxPool2d(Layer):
    """MaxPool2d layer implementation"""

    def __init__(self, kernel_size: int, stride: Optional[int] = None):
        """
        Initialize MaxPool2d layer

        Args:
            kernel_size: Size of the pooling window
            stride: Stride of the pooling operation. If None, same as kernel_size
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size

        # Cache for backward pass
        self.x = None
        self.max_indices = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of max pooling

        Args:
            x: Input of shape (N, C, H, W)

        Returns:
            Output of shape (N, C, H', W')
        """
        self.x = x
        N, C, H, W = x.shape

        # Calculate output dimensions
        out_h = (H - self.kernel_size) // self.stride + 1
        out_w = (W - self.kernel_size) // self.stride + 1

        # Initialize output and max indices
        out = np.zeros((N, C, out_h, out_w))
        self.max_indices = np.zeros_like(out, dtype=np.int32)

        # Perform max pooling
        for n in range(N):
            for c in range(C):
                for h in range(out_h):
                    for w in range(out_w):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size

                        window = x[n, c, h_start:h_end, w_start:w_end]
                        out[n, c, h, w] = np.max(window)
                        # Store the index of maximum value for backward pass
                        self.max_indices[n, c, h, w] = np.argmax(window.reshape(-1))

        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backward pass of max pooling

        Args:
            dout: Upstream derivatives

        Returns:
            Gradient with respect to input x
        """
        N, C, H, W = self.x.shape
        dx = np.zeros_like(self.x)

        # Calculate output dimensions
        out_h = (H - self.kernel_size) // self.stride + 1
        out_w = (W - self.kernel_size) // self.stride + 1

        # Distribute gradients to max elements
        for n in range(N):
            for c in range(C):
                for h in range(out_h):
                    for w in range(out_w):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size

                        # Get the index of the maximum value
                        max_idx = self.max_indices[n, c, h, w]
                        h_idx = h_start + max_idx // self.kernel_size
                        w_idx = w_start + max_idx % self.kernel_size

                        # Assign gradient to the maximum element
                        dx[n, c, h_idx, w_idx] += dout[n, c, h, w]

        return dx