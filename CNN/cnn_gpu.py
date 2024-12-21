import cupy as cp
import numpy as np
from Layers import Flatten, MaxPool2d
from BatchNorm2d import BatchNorm2d
from LossFunctions import CrossEntropyLoss
from Optimizers import Adam
from CNNData_loader import DataLoader
from Visualizer import Visualizer, FilterVisualizer
from LossFunctions import L2RegularizationLoss
from Filters import get_filters_for_layer
from DataAugmentation import HorizontalFlip, RandomRotation, RandomBrightness, Compose


# GPU-optimized Conv2D layer
class Conv2dGPU:
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        # Initialize weights on GPU
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.W = cp.asarray(np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale)
        self.b = cp.zeros(out_channels)

        self.dW = None
        self.db = None
        self.x = None
        self.x_cols = None

    def forward(self, x):
        self.x = x
        N, C, H, W = x.shape

        # Add padding if needed
        if self.padding > 0:
            x_padded = cp.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                              mode='constant')
        else:
            x_padded = x

        out_h = H - self.kernel_size + 2 * self.padding + 1
        out_w = W - self.kernel_size + 2 * self.padding + 1

        # Prepare columns
        x_cols = cp.lib.stride_tricks.as_strided(x_padded,
                                                 shape=(N, C, self.kernel_size, self.kernel_size, out_h, out_w),
                                                 strides=(x_padded.strides[0], x_padded.strides[1],
                                                          x_padded.strides[2], x_padded.strides[3],
                                                          x_padded.strides[2], x_padded.strides[3]))

        self.x_cols = x_cols.transpose(1, 2, 3, 0, 4, 5).reshape(C * self.kernel_size * self.kernel_size, -1)

        # Compute convolution using matrix multiplication
        res = cp.dot(self.W.reshape(self.out_channels, -1), self.x_cols)
        res = res.reshape(self.out_channels, N, out_h, out_w)
        res = res.transpose(1, 0, 2, 3)

        return res + self.b.reshape(1, -1, 1, 1)

    def backward(self, dout):
        N, C, H, W = self.x.shape

        # Compute gradients
        self.db = cp.sum(dout, axis=(0, 2, 3))

        dout_reshaped = dout.transpose(1, 0, 2, 3).reshape(self.out_channels, -1)
        self.dW = cp.dot(dout_reshaped, self.x_cols.T).reshape(self.W.shape)

        dx_cols = cp.dot(self.W.reshape(self.out_channels, -1).T, dout_reshaped)
        dx = cp.zeros_like(self.x)

        # Reshape back to image
        dx_cols_reshaped = dx_cols.reshape(C, self.kernel_size, self.kernel_size, N, H - self.kernel_size + 1,
                                           W - self.kernel_size + 1)
        dx_cols_reshaped = dx_cols_reshaped.transpose(3, 0, 1, 2, 4, 5)

        # Handle padding
        if self.padding > 0:
            dx = dx[:, :, self.padding:-self.padding, self.padding:-self.padding]

        return dx


class LinearGPU:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights on GPU
        scale = np.sqrt(2.0 / in_features)
        self.W = cp.asarray(np.random.randn(out_features, in_features) * scale)
        self.b = cp.zeros(out_features)

        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        return cp.dot(self.W, x) + self.b.reshape(-1, 1)

    def backward(self, dout):
        self.dW = cp.dot(dout, self.x.T)
        self.db = cp.sum(dout, axis=1)
        return cp.dot(self.W.T, dout)


class CNNGPU:
    def __init__(self):
        self.conv1 = Conv2dGPU(in_channels=3, out_channels=23, kernel_size=3, padding=1)
        self.conv1.W = cp.asarray(get_filters_for_layer(0, ['vertical_edges', 'horizontal_edges', 'diagonal1_edges',
                                                            'diagonal2_edges', 'emboss', 'sharpen', 'blur_gaussian',
                                                            'motion_blur',
                                                            'sobel_x', 'sobel_y', 'prewitt_x', 'prewitt_y', 'laplacian',
                                                            'unsharp_mask',
                                                            'median_blur', 'bilateral_blur', 'wiener_filter',
                                                            'line_filter_0_degrees',
                                                            'line_filter_45_degrees', 'line_filter_90_degrees',
                                                            'line_filter_135_degrees', 'hog_filter',
                                                            'color_histogram']))
        self.bn1 = BatchNorm2d(23)
        self.pool1 = MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = Conv2dGPU(in_channels=23, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = BatchNorm2d(32)
        self.pool2 = MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = Conv2dGPU(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = BatchNorm2d(64)
        self.pool3 = MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = Conv2dGPU(in_channels=64, out_channels=104, kernel_size=3, padding=1)
        self.bn4 = BatchNorm2d(104)
        self.pool4 = MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = Conv2dGPU(in_channels=104, out_channels=256, kernel_size=3, padding=1)
        self.bn5 = BatchNorm2d(256)

        self.flatten = Flatten()
        self.fc1 = LinearGPU(in_features=256 * 2 * 2, out_features=512)
        self.fc2 = LinearGPU(in_features=512, out_features=10)

        # Loss functions and optimizer
        self.criterion = CrossEntropyLoss()
        self.l2_reg = L2RegularizationLoss()
        self.optimizer = Adam(beta1=0.9, beta2=0.999)

        self.lambda_reg = 0.0001

        # Initialize optimizer with GPU parameters
        self.parameters = {
            'conv1_w': self.conv1.W, 'conv1_b': self.conv1.b,
            'conv2_w': self.conv2.W, 'conv2_b': self.conv2.b,
            'conv3_w': self.conv3.W, 'conv3_b': self.conv3.b,
            'conv4_w': self.conv4.W, 'conv4_b': self.conv4.b,
            'conv5_w': self.conv5.W, 'conv5_b': self.conv5.b,
            'fc1_w': self.fc1.W, 'fc1_b': self.fc1.b,
            'fc2_w': self.fc2.W, 'fc2_b': self.fc2.b,
            'bn1_gamma': cp.asarray(self.bn1.gamma),
            'bn1_beta': cp.asarray(self.bn1.beta),
            'bn2_gamma': cp.asarray(self.bn2.gamma),
            'bn2_beta': cp.asarray(self.bn2.beta),
            'bn3_gamma': cp.asarray(self.bn3.gamma),
            'bn3_beta': cp.asarray(self.bn3.beta),
            'bn4_gamma': cp.asarray(self.bn4.gamma),
            'bn4_beta': cp.asarray(self.bn4.beta),
            'bn5_gamma': cp.asarray(self.bn5.gamma),
            'bn5_beta': cp.asarray(self.bn5.beta)
        }

        self.optimizer.initialize(self.parameters)

    def forward(self, x, training=True):
        # Move input to GPU
        x = cp.asarray(x)
        self.activations = {}

        # Normalize input
        x = (x - cp.mean(x, axis=(0, 2, 3), keepdims=True)) / (cp.std(x, axis=(0, 2, 3), keepdims=True) + 1e-5)

        # Conv blocks with GPU operations
        x = self.conv1.forward(x)
        self.activations['conv1'] = x
        x = self.bn1.forward(cp.asnumpy(x), training)  # BatchNorm still on CPU
        x = cp.asarray(x)
        self.activations['bn1'] = x
        x = cp.maximum(0, x)  # ReLU
        self.activations['relu1'] = x
        x = self.pool1.forward(cp.asnumpy(x))  # MaxPool still on CPU
        x = cp.asarray(x)
        self.activations['pool1'] = x

        # Similar pattern for other layers...
        # (Repeating the same pattern for conv2-5 blocks)

        # Fully connected layers
        x = self.flatten.forward(cp.asnumpy(x))  # Flatten still on CPU
        x = cp.asarray(x)
        self.activations['flatten'] = x
        x = x.T
        x = self.fc1.forward(x)
        self.activations['fc1'] = x
        x = cp.maximum(0, x)  # ReLU
        self.activations['relu6'] = x

        x = self.fc2.forward(x)
        self.activations['fc2'] = x

        # Softmax with numerical stability
        x = x - cp.max(x, axis=0, keepdims=True)
        exp_scores = cp.exp(x)
        probs = exp_scores / cp.sum(exp_scores, axis=0, keepdims=True)
        self.activations['softmax'] = probs

        return cp.asnumpy(probs)  # Convert back to CPU for loss calculation


class CNNTrainerGPU:
    def __init__(self, model, initial_lr=1e-3, batch_size=64, lr_schedule=None):
        self.model = model
        self.initial_lr = initial_lr
        self.batch_size = batch_size
        self.lr_schedule = lr_schedule or [
            (50, 1.0),
            (30, 0.1),
            (20, 0.01)
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

    def train(self, x_train, y_train, x_val, y_val, x_test, y_test):
        # Move data to GPU
        x_train = cp.asarray(x_train)
        y_train = cp.asarray(y_train)
        x_val = cp.asarray(x_val)
        y_val = cp.asarray(y_val)
        x_test = cp.asarray(x_test)
        y_test = cp.asarray(y_test)

        # Training logic remains similar but uses GPU data
        # ... (rest of the training code remains similar but uses CuPy operations)

        return (self.train_losses, self.train_accuracies,
                self.val_losses, self.val_accuracies,
                self.test_losses, self.test_accuracies)


# Example usage
if __name__ == "__main__":
    # Load data
    x_train, y_train, x_val, y_val, x_test, y_test = DataLoader.load_cifar10_data(total_images=10000)

    # Create GPU model
    model = CNNGPU()

    # Create trainer
    trainer = CNNTrainerGPU(
        model=model,
        initial_lr=1e-3,
        batch_size=32,
        lr_schedule=[
            (1, 1.0),
            (1, 0.8),
            (1, 0.4),
            (1, 0.01)
        ]
    )

    # Train model
    results = trainer.train(x_train, y_train, x_val, y_val, x_test, y_test)

    # Visualize results
    Visualizer.plot_results(*results)