import numpy as np
from scipy.ndimage import rotate, zoom



class DataAugmentation:
    """Base class for all data augmentation techniques"""

    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, images):
        if np.random.random() < self.probability:
            return self.apply(images)
        return images

    def apply(self, images):
        raise NotImplementedError("Subclasses must implement apply method")


class HorizontalFlip(DataAugmentation):
    """Horizontally flip the images"""

    def apply(self, images):
        return images[:, :, :, ::-1]


class VerticalFlip(DataAugmentation):
    """Vertically flip the images"""

    def apply(self, images):
        return images[:, :, ::-1, :]


class RandomRotation(DataAugmentation):
    """Rotate images by a random angle within given bounds"""

    def __init__(self, angle_range=(-15, 15), probability=0.5):
        super().__init__(probability)
        self.angle_range = angle_range

    def apply(self, images):
        angle = np.random.uniform(*self.angle_range)
        rotated = np.zeros_like(images)
        for i in range(len(images)):
            for c in range(images.shape[1]):  # For each channel
                rotated[i, c] = rotate(images[i, c], angle, reshape=False)
        return rotated


class RandomZoom(DataAugmentation):
    """Randomly zoom in/out on images"""

    def __init__(self, zoom_range=(0.8, 1.2), probability=0.5):
        super().__init__(probability)
        self.zoom_range = zoom_range

    def apply(self, images):
        zoom_factor = np.random.uniform(*self.zoom_range)
        zoomed = np.zeros_like(images)
        for i in range(len(images)):
            for c in range(images.shape[1]):  # For each channel
                zoomed[i, c] = zoom(images[i, c], zoom_factor)
        return zoomed


class RandomBrightness(DataAugmentation):
    """Randomly adjust image brightness"""

    def __init__(self, brightness_range=(0.8, 1.2), probability=0.5):
        super().__init__(probability)
        self.brightness_range = brightness_range

    def apply(self, images):
        factor = np.random.uniform(*self.brightness_range)
        return np.clip(images * factor, 0, 1)


class RandomContrast(DataAugmentation):
    """Randomly adjust image contrast"""

    def __init__(self, contrast_range=(0.8, 1.2), probability=0.5):
        super().__init__(probability)
        self.contrast_range = contrast_range

    def apply(self, images):
        factor = np.random.uniform(*self.contrast_range)
        mean = np.mean(images, axis=(2, 3), keepdims=True)
        return np.clip(mean + (images - mean) * factor, 0, 1)


class GaussianNoise(DataAugmentation):
    """Add random Gaussian noise to images"""

    def __init__(self, std_range=(0, 0.05), probability=0.5):
        super().__init__(probability)
        self.std_range = std_range

    def apply(self, images):
        std = np.random.uniform(*self.std_range)
        noise = np.random.normal(0, std, images.shape)
        return np.clip(images + noise, 0, 1)

class Compose:
    """Compose multiple augmentation techniques"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images):
        for transform in self.transforms:
            images = transform(images)
        return images

