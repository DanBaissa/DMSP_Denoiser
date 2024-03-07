import numpy as np
import rasterio
from rasterio.plot import show
from tensorflow.keras.models import load_model
from scipy.ndimage import zoom
from pathlib import Path
import matplotlib.pyplot as plt

# Load the saved model
model = load_model('Test_data/Germany_trained_model.h5')

# Path to your raster file
raster_path = 'Test_data/cropped_F18_20131201_20131231.cloud2.light1.marginal0.glare2.line_screened.avg_vis.tif'

# Open the raster file
with rasterio.open(raster_path) as raster:
    # Plot the raster
    fig, ax = plt.subplots(figsize=(10, 10))
    show(raster, ax=ax)

    # Display dimensions on the plot
    ax.set_title(f"Raster Dimensions: {raster.width}x{raster.height}")
    plt.show()

# def resize_and_pad(arr):
#     # Your provided resizing and padding logic
#     target_size_largest = 2 ** np.floor(np.log2(max(arr.shape)))
#     smaller_dim = min(arr.shape)
#     target_size_smaller = 2 ** np.ceil(np.log2(smaller_dim))
#     resize_ratio = target_size_largest / max(arr.shape)
#     new_shape = (int(arr.shape[0] * resize_ratio), int(arr.shape[1] * resize_ratio))
#     resized_arr = zoom(arr, (new_shape[0] / arr.shape[0], new_shape[1] / arr.shape[1]), order=1)
#
#     if arr.shape[0] < arr.shape[1]:  # Height is the smaller dimension
#         pad_height = (int(target_size_smaller) - resized_arr.shape[0]) // 2
#         pad_width = 0
#     else:  # Width is the smaller dimension
#         pad_width = (int(target_size_smaller) - resized_arr.shape[1]) // 2
#         pad_height = 0
#
#     padded_arr = np.pad(resized_arr, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant',
#                         constant_values=0)
#     return padded_arr
#
