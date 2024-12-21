#Visualizer.py
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

class Visualizer:
    @staticmethod
    def plot_results(train_losses, train_accuracies, val_losses, val_accuracies, test_losses, test_accuracies):
        plt.figure(figsize=(12, 8))

        plt.subplot(221)
        plt.plot(train_losses, label='Train')
        plt.plot(val_losses, label='Validation Loss')
        plt.plot(test_losses, label='Test')
        plt.xlabel('Iterations (x10)')
        plt.ylabel('Loss')
        plt.title('Loss vs Iterations')
        plt.legend()

        plt.subplot(222)
        plt.plot(train_accuracies, label='Train')
        plt.plot(val_accuracies, label='Validation Loss')
        plt.plot(test_accuracies, label='Test')
        plt.xlabel('Iterations (x10)')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Iterations')
        plt.legend()

        plt.tight_layout()
        plt.show()


class FilterVisualizer:
    @staticmethod
    def visualize_filters(network, save_path=None):
        """
        Visualize all filters from all convolutional layers of any CNN.
        Creates a separate figure for each layer showing all its filters.

        Args:
            network: Neural network object
            save_path (str, optional): Base path to save the visualizations
                                     Will append _layerN.png for each layer
        """
        # Find all convolutional layers in the network
        conv_layers = {}
        for attr_name in dir(network):
            attr = getattr(network, attr_name)
            # Check if attribute has weights (W) and they are 4D (typical for conv layers)
            if hasattr(attr, 'W') and isinstance(attr.W, np.ndarray) and len(attr.W.shape) == 4:
                conv_layers[attr_name] = attr.W

        if not conv_layers:
            raise ValueError("No convolutional layers found in the network")

        # Visualize each layer separately
        for layer_idx, (layer_name, filters) in enumerate(conv_layers.items()):
            FilterVisualizer._visualize_layer(filters, layer_name, layer_idx, save_path)

    @staticmethod
    def _visualize_layer(filters, layer_name, layer_idx, save_path=None):
        """Visualize all filters in a single layer"""
        n_filters = filters.shape[0]  # Get total number of filters
        kernel_size = filters.shape[2]

        # Calculate grid dimensions
        grid_size = int(np.ceil(np.sqrt(n_filters)))

        # Create figure
        fig = plt.figure(figsize=(20, 20))
        plt.suptitle(f'{layer_name} Filters (Total: {n_filters})', fontsize=16)

        # Create image grid
        grid = ImageGrid(fig, 111,
                         nrows_ncols=(grid_size, grid_size),
                         axes_pad=0.3,
                         share_all=True)

        # Prepare and plot each filter
        for idx in range(n_filters):
            ax = grid[idx]

            if layer_idx == 0:  # First layer (RGB)
                # Handle RGB filters
                f = filters[idx].transpose(1, 2, 0)
                if f.shape[-1] == 1:  # Grayscale
                    f = np.repeat(f, 3, axis=-1)
                # Normalize for visualization
                f = (f - f.min()) / (f.max() - f.min() + 1e-8)
                ax.imshow(f)
            else:
                # For deeper layers, show averaged patterns
                f = np.mean(filters[idx], axis=0)
                f = (f - f.min()) / (f.max() - f.min() + 1e-8)
                ax.imshow(f, cmap='viridis')

            ax.axis('off')
            ax.set_title(f'Filter {idx + 1}')

        # Adjust layout and save/show
        plt.tight_layout()

        if save_path:
            # Create unique filename for each layer
            base_path = save_path.rsplit('.', 1)[0]  # Remove extension if present
            extension = save_path.rsplit('.', 1)[1] if '.' in save_path else 'png'
            layer_save_path = f"{base_path}_layer{layer_idx + 1}.{extension}"
            plt.savefig(layer_save_path, bbox_inches='tight', dpi=300)
            print(f"Saved visualization for {layer_name} to {layer_save_path}")

        plt.show()

    @staticmethod
    def visualize_filter_responses(network, input_image, save_path=None):
        """
        Visualize the response maps of all filters for a given input image.

        Args:
            network: Neural network object
            input_image: Input image to visualize filter responses for
            save_path (str, optional): Base path to save the visualizations
        """
        # Get activations after forward pass
        _ = network.forward(input_image[np.newaxis, ...], training=False)

        # Visualize activations for each conv layer
        for layer_name, activation in network.activations.items():
            if layer_name.startswith('conv'):
                FilterVisualizer._visualize_activation_maps(
                    activation[0],  # Remove batch dimension
                    layer_name,
                    save_path
                )

    @staticmethod
    def _visualize_activation_maps(activation_maps, layer_name, save_path=None):
        """Visualize activation maps for a single layer"""
        n_maps = activation_maps.shape[0]
        grid_size = int(np.ceil(np.sqrt(n_maps)))

        fig = plt.figure(figsize=(20, 20))
        plt.suptitle(f'{layer_name} Activation Maps (Total: {n_maps})', fontsize=16)

        grid = ImageGrid(fig, 111,
                         nrows_ncols=(grid_size, grid_size),
                         axes_pad=0.3,
                         share_all=True)

        for idx in range(n_maps):
            ax = grid[idx]
            feature_map = activation_maps[idx]

            # Normalize for visualization
            feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
            ax.imshow(feature_map, cmap='viridis')
            ax.axis('off')
            ax.set_title(f'Map {idx + 1}')

        plt.tight_layout()

        if save_path:
            base_path = save_path.rsplit('.', 1)[0]
            extension = save_path.rsplit('.', 1)[1] if '.' in save_path else 'png'
            activation_save_path = f"{base_path}_{layer_name}_activations.{extension}"
            plt.savefig(activation_save_path, bbox_inches='tight', dpi=300)
            print(f"Saved activation maps for {layer_name} to {activation_save_path}")

        plt.show()