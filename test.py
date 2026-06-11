# # test if the CUDA_ERROR_INVALID_HANDLE is due to libraries or gpu
# #  I initialised a separate environment and installed tensorflow separately in it so this code used that environment 

# import tensorflow as tf

# a = tf.constant([1,2])
# b = tf.constant([3,4])

# print(a + b)

# # results the above code runs so the issue is jac

# another test with tensorflow using the official docs: https://www.tensorflow.org/install/gpu

import tensorflow as tf

print(tf.__version__)
print(tf.config.list_physical_devices("GPU"))

a = tf.constant([1.0, 2.0])
b = tf.constant([2.0, 4.0])

print(a + b)