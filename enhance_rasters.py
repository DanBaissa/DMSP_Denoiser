import json
import rasterio
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
from tensorflow.keras.models import load_model

# Load metadata to get input_shape
metadata_path = 'Test_data/Germany_trained_model_metadata.json'
with open(metadata_path, 'r') as file:
    metadata = json.load(file)
    target_height, target_width, _ = metadata['input_shape']

# Load the model
model_path = 'Test_data/Germany_trained_model.h5'
model = load_model(model_path)

# Load and resize the raster data
raster_path = 'Test_data/cropped_F18_20131201_20131231.cloud2.light1.marginal0.glare2.line_screened.avg_vis.tif'
# Load and resize the raster data
with rasterio.open(raster_path) as raster:
    data = raster.read(1)  # Read the first band
    resized_data = resize(data, (target_height, target_width), anti_aliasing=True)

# Normalize the resized raster data using its maximum value
max_value = np.max(resized_data)
if max_value > 0:  # Prevent division by zero
    normalized_data = resized_data / max_value
else:
    normalized_data = resized_data  # Handle the case where the max value is 0

# Model prediction
# Assuming the model expects data in the shape of (1, height, width, channels)
input_data = np.expand_dims(normalized_data, axis=[0, -1])  # Add batch and channel dimensions
prediction = model.predict(input_data)

# Assuming the model outputs data in the shape of (1, height, width, channels)
predicted_image = prediction[0, :, :, 0]  # Remove batch dimension and select the first channel

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(20, 10))

# Plot original resized raster
axs[0].imshow(resized_data, cmap='gray')
axs[0].set_title('Original DMSP Raster')

# Plot predicted output
axs[1].imshow(predicted_image, cmap='gray')
axs[1].set_title('Model Output')

# Save to PDF
plt.savefig('raster_and_prediction.pdf')
plt.show()
