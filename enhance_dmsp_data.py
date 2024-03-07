import rasterio
import numpy as np
from tensorflow.keras.models import load_model
from skimage.transform import resize

def read_expected_shape(shape_path="expected_shape.txt"):
    with open(shape_path, "r") as file:
        shape_str = file.readline().strip().split()
        return tuple(map(int, shape_str))

def load_and_preprocess_raster(raster_path, expected_shape):
    with rasterio.open(raster_path) as src:
        raster_data = src.read(1)  # Read the first band
        resized_data = resize(raster_data, expected_shape, preserve_range=True, anti_aliasing=True).astype(np.float32)
    return resized_data, src.meta

def enhance_raster(model_path, raster_data):
    model = load_model(model_path)
    enhanced_data = model.predict(raster_data[np.newaxis, ..., np.newaxis])
    return enhanced_data.squeeze()

def save_enhanced_raster(enhanced_data, meta, save_path):
    with rasterio.open(save_path, 'w', **meta) as dst:
        dst.write(enhanced_data, 1)

def enhance_dmsp_data(model_path, dmsp_raster_path, save_path):
    expected_shape = read_expected_shape()
    raster_data, meta = load_and_preprocess_raster(dmsp_raster_path, expected_shape)
    enhanced_data = enhance_raster(model_path, raster_data)
    save_enhanced_raster(enhanced_data, meta, save_path)
    print(f"Enhanced raster saved to {save_path}")
