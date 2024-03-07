import numpy as np
import rasterio
from tensorflow.keras.models import load_model
from scipy.ndimage import zoom
from pathlib import Path


def resize_and_pad(arr):
    # Your provided resizing and padding logic
    target_size_largest = 2 ** np.floor(np.log2(max(arr.shape)))
    smaller_dim = min(arr.shape)
    target_size_smaller = 2 ** np.ceil(np.log2(smaller_dim))
    resize_ratio = target_size_largest / max(arr.shape)
    new_shape = (int(arr.shape[0] * resize_ratio), int(arr.shape[1] * resize_ratio))
    resized_arr = zoom(arr, (new_shape[0] / arr.shape[0], new_shape[1] / arr.shape[1]), order=1)

    if arr.shape[0] < arr.shape[1]:  # Height is the smaller dimension
        pad_height = (int(target_size_smaller) - resized_arr.shape[0]) // 2
        pad_width = 0
    else:  # Width is the smaller dimension
        pad_width = (int(target_size_smaller) - resized_arr.shape[1]) // 2
        pad_height = 0

    padded_arr = np.pad(resized_arr, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant',
                        constant_values=0)
    return padded_arr


def enhance_rasters(dmsp_dir, selected_country, bin_factor, output_folder, model_file):
    model = load_model(model_file)

    dmsp_raster_paths = list(Path(dmsp_dir).glob('*.tif'))
    for dmsp_raster_path in dmsp_raster_paths:
        with rasterio.open(dmsp_raster_path) as src:
            dmsp_data = src.read(1).astype(np.float32)

            # Apply binning (implement your binning logic as needed)

            # Resize and pad the data
            dmsp_data_prepared = resize_and_pad(dmsp_data)

            # Add batch dimension
            dmsp_data_batch = np.expand_dims(dmsp_data_prepared, axis=0)

            # Enhance the raster using the model
            enhanced_data = model.predict(dmsp_data_batch)

            # Remove batch dimension and post-process if necessary
            enhanced_data_processed = np.squeeze(enhanced_data)

            # Save the enhanced raster
            enhanced_raster_path = Path(output_folder) / f"enhanced_{dmsp_raster_path.stem}.tif"
            save_enhanced_raster(enhanced_data_processed, src.profile, enhanced_raster_path)


def save_enhanced_raster(enhanced_data, profile, save_path):
    profile.update({
        'height': enhanced_data.shape[0],
        'width': enhanced_data.shape[1],
        'dtype': 'float32'
    })

    with rasterio.open(save_path, 'w', **profile) as dst:
        dst.write(enhanced_data, 1)

    print(f"Enhanced raster saved to {save_path}")
