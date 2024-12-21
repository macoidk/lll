import numpy as np


def vertical_edges():
    """Edge detection filter for vertical edges"""
    return np.array([[-1, -1, -1],
                     [2, 2, 2],
                     [-1, -1, -1]])


def horizontal_edges():
    """Edge detection filter for horizontal edges"""
    return np.array([[-1, 2, -1],
                     [-1, 2, -1],
                     [-1, 2, -1]])


def diagonal1_edges():
    """Edge detection filter for diagonal edges (top-left to bottom-right)"""
    return np.array([[-1, -1, 2],
                     [-1, 2, -1],
                     [2, -1, -1]])


def diagonal2_edges():
    """Edge detection filter for diagonal edges (top-right to bottom-left)"""
    return np.array([[2, -1, -1],
                     [-1, 2, -1],
                     [-1, -1, 2]])


def emboss():
    """Emboss filter"""
    return np.array([[-2, -1, 0],
                     [-1, 1, 1],
                     [0, 1, 2]])


def sharpen():
    """Sharpen filter"""
    return np.array([[0, -1, 0],
                     [-1, 5, -1],
                     [0, -1, 0]])


def blur_gaussian(kernel_size=5, sigma=1.0):
    """
    Create a Gaussian blur filter.

    Args:
        kernel_size (int): Size of the Gaussian kernel.
        sigma (float): Standard deviation of the Gaussian kernel.

    Returns:
        numpy.ndarray: Gaussian kernel of shape (kernel_size, kernel_size).
    """
    center = kernel_size // 2
    kernel = np.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel /= (2 * np.pi * sigma ** 2)
    return kernel

def median_blur(kernel_size=3):
    """
    Create a median blur filter.

    Args:
        kernel_size (int): Size of the median kernel.

    Returns:
        numpy.ndarray: Median blur kernel of shape (kernel_size, kernel_size).
    """
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    return kernel

def bilateral_blur(kernel_size=5, sigma_color=10.0, sigma_space=5.0):
    """
    Create a bilateral blur filter.

    Args:
        kernel_size (int): Size of the bilateral kernel.
        sigma_color (float): Standard deviation of the color space Gaussian kernel.
        sigma_space (float): Standard deviation of the spatial Gaussian kernel.

    Returns:
        numpy.ndarray: Bilateral blur kernel of shape (kernel_size, kernel_size).
    """
    center = kernel_size // 2
    kernel = np.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma_space ** 2)) * \
                           np.exp(-(x ** 2 + y ** 2) / (2 * sigma_color ** 2))
    kernel /= kernel.sum()
    return kernel

def wiener_filter(kernel_size=5, noise_variance=0.01):
    """
    Create an adaptive Wiener filter.

    Args:
        kernel_size (int): Size of the Wiener kernel.
        noise_variance (float): Variance of the noise.

    Returns:
        numpy.ndarray: Wiener filter kernel of shape (kernel_size, kernel_size).
    """
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    return kernel / (kernel ** 2 + noise_variance)

def unsharp_mask(alpha=0.2):
    """
    Create an unsharp mask filter.

    Args:
        alpha (float): Sharpening factor.

    Returns:
        numpy.ndarray: Unsharp mask kernel of shape (3, 3).
    """
    return np.array([[-alpha, -alpha, -alpha],
                     [-alpha, 1 + 4 * alpha, -alpha],
                     [-alpha, -alpha, -alpha]])


def motion_blur(kernel_size=9, angle=0):
    """
    Create a motion blur filter.

    Args:
        kernel_size (int): Size of the motion blur kernel.
        angle (float): Angle of the motion blur in degrees.

    Returns:
        numpy.ndarray: Motion blur kernel of shape (kernel_size, kernel_size).
    """
    kernel = np.zeros((kernel_size, kernel_size))
    angle_rad = np.deg2rad(angle)
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - (kernel_size - 1) / 2
            y = j - (kernel_size - 1) / 2
            kernel[i, j] = np.exp(-((x * np.cos(angle_rad) + y * np.sin(angle_rad)) ** 2 +
                                    (-x * np.sin(angle_rad) + y * np.cos(angle_rad)) ** 2) /
                                  (2 * (kernel_size / 2) ** 2))
    kernel /= kernel.sum()
    return kernel


def sobel_x():
    """Sobel filter for detecting edges in the x-direction"""
    return np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]])


def sobel_y():
    """Sobel filter for detecting edges in the y-direction"""
    return np.array([[-1, -2, -1],
                     [0, 0, 0],
                     [1, 2, 1]])


def prewitt_x():
    """Prewitt filter for detecting edges in the x-direction"""
    return np.array([[-1, 0, 1],
                     [-1, 0, 1],
                     [-1, 0, 1]])


def prewitt_y():
    """Prewitt filter for detecting edges in the y-direction"""
    return np.array([[-1, -1, -1],
                     [0, 0, 0],
                     [1, 1, 1]])


def laplacian():
    """Laplacian filter for detecting edges"""
    return np.array([[0, 1, 0],
                     [1, -4, 1],
                     [0, 1, 0]])

def line_filter_0_degrees():
    """Line detection filter for 0 degree lines"""
    return np.array([[-1, -1, -1],
                    [2, 2, 2],
                    [-1, -1, -1]])

def line_filter_45_degrees():
    """Line detection filter for 45 degree lines"""
    return np.array([[-1, -1, 2],
                    [-1, 2, -1],
                    [2, -1, -1]])

def line_filter_90_degrees():
    """Line detection filter for 90 degree lines"""
    return np.array([[-1, 2, -1],
                    [-1, 2, -1],
                    [-1, 2, -1]])

def line_filter_135_degrees():
    """Line detection filter for 135 degree lines"""
    return np.array([[2, -1, -1],
                    [-1, 2, -1],
                    [-1, -1, 2]])

def unsharp_mask():
    """Unsharp mask filter for sharpening"""
    return np.array([[-1, -1, -1],
                     [-1, 9, -1],
                     [-1, -1, -1]]) / 9.0

def color_histogram(num_bins=8):
    """
    Create a color histogram filter.

    Args:
        num_bins (int): Number of bins for the color histogram.

    Returns:
        numpy.ndarray: Color histogram filter of shape (num_bins, num_bins, num_bins).
    """
    return np.array([[1 / 9, 1 / 9, 1 / 9],
                     [1 / 9, 1 / 9, 1 / 9],
                     [1 / 9, 1 / 9, 1 / 9]])

def hog_filter(cell_size=8, block_size=2, num_bins=9):
    """
    Create a Histogram of Oriented Gradients (HOG) filter.

    Args:
        cell_size (int): Size of the cell in pixels.
        block_size (int): Size of the block in cells.
        num_bins (int): Number of orientation bins.

    Returns:
        numpy.ndarray: HOG filter of shape (block_size * block_size * num_bins, 1, cell_size, cell_size).
    """
    return np.array([[1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]])


def get_filters_for_layer(layer_idx, filter_names):
    """
    Modified function to ensure consistent filter shapes
    """
    filters = []
    for name in filter_names:
        if name == 'vertical_edges':
            filters.append(vertical_edges())
        elif name == 'horizontal_edges':
            filters.append(horizontal_edges())
        elif name == 'diagonal1_edges':
            filters.append(diagonal1_edges())
        elif name == 'diagonal2_edges':
            filters.append(diagonal2_edges())
        elif name == 'emboss':
            filters.append(emboss())
        elif name == 'sharpen':
            filters.append(sharpen())
        elif name == 'blur_gaussian':
            kernel = blur_gaussian()
            # Ensure 3x3 shape by taking center portion if larger
            if kernel.shape[0] > 3:
                center = kernel.shape[0] // 2
                kernel = kernel[center - 1:center + 2, center - 1:center + 2]
            filters.append(kernel)
        elif name == 'motion_blur':
            kernel = motion_blur()
            # Ensure 3x3 shape by taking center portion if larger
            if kernel.shape[0] > 3:
                center = kernel.shape[0] // 2
                kernel = kernel[center - 1:center + 2, center - 1:center + 2]
            filters.append(kernel)
        elif name == 'sobel_x':
            filters.append(sobel_x())
        elif name == 'sobel_y':
            filters.append(sobel_y())
        elif name == 'prewitt_x':
            filters.append(prewitt_x())
        elif name == 'prewitt_y':
            filters.append(prewitt_y())
        elif name == 'laplacian':
            filters.append(laplacian())
        elif name == 'unsharp_mask':
            filters.append(unsharp_mask())
        elif name == 'median_blur':
            kernel = median_blur()
            if kernel.shape[0] > 3:
                center = kernel.shape[0] // 2
                kernel = kernel[center - 1:center + 2, center - 1:center + 2]
            filters.append(kernel)
        elif name == 'bilateral_blur':
            kernel = bilateral_blur()
            if kernel.shape[0] > 3:
                center = kernel.shape[0] // 2
                kernel = kernel[center - 1:center + 2, center - 1:center + 2]
            filters.append(kernel)
        elif name == 'wiener_filter':
            kernel = wiener_filter()
            if kernel.shape[0] > 3:
                center = kernel.shape[0] // 2
                kernel = kernel[center - 1:center + 2, center - 1:center + 2]
            filters.append(kernel)
        elif name == 'line_filter_0_degrees':
            filters.append(line_filter_0_degrees())
        elif name == 'line_filter_45_degrees':
            filters.append(line_filter_45_degrees())
        elif name == 'line_filter_90_degrees':
            filters.append(line_filter_90_degrees())
        elif name == 'line_filter_135_degrees':
            filters.append(line_filter_135_degrees())
        elif name == 'hog_filter':
            filters.append(hog_filter())
        elif name == 'color_histogram':
            filters.append(color_histogram())
        else:
            raise ValueError(f"Invalid filter name: {name}")

    # Convert to numpy array and reshape to (out_channels, in_channels, height, width)
    filters = np.array(filters)  # Shape: (num_filters, 3, 3)
    filters = filters[:, np.newaxis, :, :]  # Shape: (num_filters, 1, 3, 3)
    in_channels = 3
    # Repeat for each input channel and normalize
    filters = np.repeat(filters, in_channels, axis=1)  # Shape: (num_filters, in_channels, 3, 3)

    # Normalize filters
    for i in range(len(filters)):
        filters[i] = filters[i] / np.sqrt(np.sum(filters[i] ** 2) + 1e-8)

    return filters