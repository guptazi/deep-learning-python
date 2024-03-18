import numpy as np

inputs = [[0.5, 1.2, 2.3, 1.8], [1.2, 3.5, -0.8, 1.5], [-0.8, 1.4, 1.7, -0.4]]
weights = [[0.1, 0.9, -0.4, 0.7],
           [0.3, -0.8, 0.1, -0.4],
           [-0.2, -0.2, 0.3, 0.6]]
biases = [1.5, 2.5, 0.2]

weights2 = [[0.2, -0.2, 0.8],
            [-0.8, 0.2, -0.5],
            [-0.3, 0.6, -0.1]]
biases2 = [-1.5, 2.3, -0.7]

layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

print(layer2_outputs)
print(layer1_outputs)
