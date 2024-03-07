#raster_to_arrays.py
import numpy as np
import rasterio
from scipy.ndimage import zoom
from pathlib import Path


def resize_and_pad_array(arr):
    original_shape = arr.shape
    print(f"Original dimensions: {original_shape}")

    # Determine the next lowest power of two for the largest dimension
    target_size_largest = 2 ** np.floor(np.log2(max(arr.shape)))

    # Determine the next highest power of two for the smaller dimension
    smaller_dim = min(arr.shape)
    target_size_smaller = 2 ** np.ceil(np.log2(smaller_dim))

    # Calculate the resize ratio to maintain the aspect ratio
    resize_ratio = target_size_largest / max(arr.shape)
    new_shape = (int(arr.shape[0] * resize_ratio), int(arr.shape[1] * resize_ratio))

    # Resize the array using zoom while maintaining the aspect ratio
    resized_arr = zoom(arr, (new_shape[0] / arr.shape[0], new_shape[1] / arr.shape[1]), order=1)

    # Determine which dimension is the smaller one and calculate padding
    if arr.shape[0] < arr.shape[1]:  # Height is the smaller dimension
        pad_height = (int(target_size_smaller) - resized_arr.shape[0]) // 2
        pad_width = 0  # No padding needed for width
    else:  # Width is the smaller dimension
        pad_width = (int(target_size_smaller) - resized_arr.shape[1]) // 2
        pad_height = 0  # No padding needed for height

    # Apply padding
    padded_arr = np.pad(resized_arr, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant',
                        constant_values=0)

    # Print the post-correction dimensions and padding information
    post_correction_shape = padded_arr.shape
    print(f"Post-correction dimensions: {post_correction_shape}")
    if pad_height > 0:
        print("Padding added to the height dimension.")
    if pad_width > 0:
        print("Padding added to the width dimension.")

    return padded_arr


def load_cropped_rasters_to_arrays(dmsp_dir, bm_dir, bin_factor):
    X = []  # List for DMSP data arrays
    y = []  # List for BM data arrays

    # Load, bin, resize, and pad BM rasters
    for bm_raster_path in Path(bm_dir).glob('cropped_*.tif'):
        with rasterio.open(bm_raster_path) as src:
            bm_data = src.read(1).astype(np.float32)  # Read the first band and convert to float32
            if bin_factor > 1:
                bm_data = bin_array(bm_data, bin_factor)  # Bin the BM data
            bm_data_resized_padded = resize_and_pad_array(bm_data)
            # Normalize the resized and padded BM data
            bm_data_normalized = bm_data_resized_padded / np.max(bm_data_resized_padded) if np.max(bm_data_resized_padded) > 0 else bm_data_resized_padded
            y.append(bm_data_normalized.astype(np.float32))  # Ensure the normalized data is also float32

    # Load, bin, resize, and pad DMSP rasters similarly
    for dmsp_raster_path in Path(dmsp_dir).glob('cropped_*.tif'):
        with rasterio.open(dmsp_raster_path) as src:
            dmsp_data = src.read(1).astype(np.float32)  # Read the first band and convert to float32
            if bin_factor > 1:
                dmsp_data = bin_array(dmsp_data, bin_factor)  # Bin the DMSP data
            dmsp_data_resized_padded = resize_and_pad_array(dmsp_data)
            # Normalize the resized and padded DMSP data
            dmsp_data_normalized = dmsp_data_resized_padded / np.max(dmsp_data_resized_padded) if np.max(dmsp_data_resized_padded) > 0 else dmsp_data_resized_padded
            X.append(dmsp_data_normalized.astype(np.float32))  # Ensure the normalized data is also float32

    return X, y



def bin_array(arr, bin_factor):
    # Calculate the shape of the binned array
    binned_shape = (arr.shape[0] // bin_factor, arr.shape[1] // bin_factor)

    # Initialize the binned array
    binned_arr = np.zeros(binned_shape)

    # Perform binning by averaging over bin_factor x bin_factor blocks
    for i in range(binned_shape[0]):
        for j in range(binned_shape[1]):
            binned_arr[i, j] = np.mean(arr[i * bin_factor:(i + 1) * bin_factor, j * bin_factor:(j + 1) * bin_factor])

    return binned_arr
