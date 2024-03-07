import json

# Define the path to the input shape JSON file
shape_filename = 'Germany_input_shape.json'

# Load the input shape from the JSON file
with open(shape_filename, 'r') as f:
    data = json.load(f)
    input_shape = data['input_shape']
    bin_factor = data['bin_factor']

# Print the input shape size
print(f"The input shape size used for training the model: {bin_factor}")
