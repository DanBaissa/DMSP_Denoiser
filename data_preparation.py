import numpy as np
from sklearn.model_selection import train_test_split
from Raster_to_arrays import load_cropped_rasters_to_arrays  # Assuming this function is defined as discussed

def prepare_data_for_training(dmsp_dir, bm_dir, bin_factor, test_size=0.2):
    # Load and normalize the raster data
    X, y = load_cropped_rasters_to_arrays(dmsp_dir, bm_dir, bin_factor)

    # Reshape the data
    X = np.array([np.expand_dims(x, axis=-1) for x in X])
    y = np.array([np.expand_dims(y_item, axis=-1) for y_item in y])

    # Optionally split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)

    return X_train, X_val, y_train, y_val
