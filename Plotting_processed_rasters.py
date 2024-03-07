# Plotting_processed_rasters.py
import matplotlib.pyplot as plt
import numpy as np

def plot_and_save_first_rasters(X_train, X_test, y_train, y_test, save_path="raster_plots"):
    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True)

    def save_raster(raster, filename):
        plt.imsave(save_path / filename, raster[:, :, 0], cmap='gray')

    # Save the first raster in each dataset
    save_raster(X_train[0], "X_train_first_raster.png")
    save_raster(X_test[0], "X_test_first_raster.png")
    save_raster(y_train[0], "y_train_first_raster.png")
    save_raster(y_test[0], "y_test_first_raster.png")

    # Plotting the first raster of each dataset
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    axs[0, 0].imshow(X_train[0, :, :, 0], cmap='gray')
    axs[0, 0].set_title('First Raster in X_train')
    axs[0, 0].axis('off')

    axs[0, 1].imshow(X_test[0, :, :, 0], cmap='gray')
    axs[0, 1].set_title('First Raster in X_test')
    axs[0, 1].axis('off')

    axs[1, 0].imshow(y_train[0, :, :, 0], cmap='gray')
    axs[1, 0].set_title('First Raster in y_train')
    axs[1, 0].axis('off')

    axs[1, 1].imshow(y_test[0, :, :, 0], cmap='gray')
    axs[1, 1].set_title('First Raster in y_test')
    axs[1, 1].axis('off')

    plt.tight_layout()
    plt.show()
