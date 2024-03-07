#main.py
import tensorflow as tf
import tkinter as tk
from gui import SatelliteGUI
import geopandas as gpd
from preprocess import preprocess_rasters
from data_preparation import prepare_data_for_training
from model import create_model
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import json
import os

def load_countries(shapefile_path="World_Countries/World_Countries_Generalized.shp"):
    data = gpd.read_file(shapefile_path)
    return sorted(data['COUNTRY'].unique().tolist())

def train_model(dmsp_folder, bm_folder, selected_country, conv_size, bin_factor, save_path):
    print("Model training initiated with conv_size =", conv_size, "and bin_factor =", bin_factor)
    try:
        # Step 1: Preprocess the Data
        preprocess_data(dmsp_folder, bm_folder, selected_country)

        # Step 2: Load and Prepare the Data
        X_train, X_val, y_train, y_val = prepare_data_for_training(dmsp_folder, bm_folder, bin_factor)

        # Convert X and y to float32
        X_train = np.array(X_train, dtype=np.float32)
        X_val = np.array(X_val, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32)
        y_val = np.array(y_val, dtype=np.float32)

        # Step 3: Define the Model
        model = create_model(conv_size)  # Adjust conv_size as needed

        # Step 4: Train the Model
        # Define early stopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)

        # Define model checkpoint callback
        model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min',
                                           verbose=1)

        # Fit the model with callbacks
        history = model.fit(X_train, y_train, epochs=1000,
                            # Set a high number since EarlyStopping will take care of stopping
                            batch_size=32, validation_data=(X_val, y_val),
                            callbacks=[early_stopping, model_checkpoint])

        # Step 5: Save the Model
        input_shape = X_train.shape[1:]  # Excludes the batch size dimension

        if save_path:
            model_filename = os.path.join(save_path, f"{selected_country}_trained_model.h5")
            metadata_filename = os.path.join(save_path, f"{selected_country}_trained_model_metadata.json")
        else:
            model_filename = f"{selected_country}_trained_model.h5"
            metadata_filename = f"{selected_country}_trained_model_metadata.json"

        # Step 5: Save the Model
        model.save(model_filename)
        print(f"Model saved successfully as {model_filename}.")

        # Save the input shape and bin_factor to a JSON file
        metadata = {
            'input_shape': input_shape,
            'bin_factor': bin_factor
        }
        with open(metadata_filename, 'w') as f:
            json.dump(metadata, f)
            print(f"Metadata including input shape and binning factor saved successfully as {metadata_filename}.")

        tk.messagebox.showinfo("Success", f"Model trained and saved as {model_filename}.")
    except Exception as e:
        tk.messagebox.showerror("Error", str(e))
        print(e)


def preprocess_data(dmsp_folder, bm_folder, selected_country):
    try:
        preprocess_rasters(dmsp_folder, bm_folder, selected_country)
        tk.messagebox.showinfo("Success", "Data preprocessing completed successfully.")
    except Exception as e:
        tk.messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    # Check if TensorFlow is using the GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("Using GPU devices: ", gpus)
    else:
        print("No GPU devices available.")

    root = tk.Tk()
    countries = load_countries()
    gui = SatelliteGUI(root, countries,
                       lambda dmsp_folder, bm_folder, selected_country, conv_size=3, bin_factor=4,
                              save_path=None: train_model(
                           dmsp_folder, bm_folder, selected_country, conv_size, bin_factor, save_path))

    root.mainloop()
