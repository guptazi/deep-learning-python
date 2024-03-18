import numpy as np
inputs = [2.0, 5.0, 8.0, 9.5]
weights = [0.6, 0.5, -0.4, 4.0]
bias = 5.0
outputs = np.dot(weights, inputs) + bias
print(outputs)
