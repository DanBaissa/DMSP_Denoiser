import tensorflow as tf

# Check if TensorFlow can access the GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("Using GPU devices: ", gpus)
else:
    print("No GPU devices available.")


print("TensorFlow version:", tf.__version__)
print("Is GPU available?", tf.test.is_gpu_available())
print("Is TensorFlow built with CUDA?", tf.test.is_built_with_cuda())
print("CUDA version:", tf.sysconfig.get_build_info()["cuda_version"])
print("cuDNN version:", tf.sysconfig.get_build_info()["cudnn_version"])


