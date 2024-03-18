#ReLU Activation
import numpy as np

class ReLU:
# Forward pass
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Softmax:
    # Forward pass
    def forward(self, inputs):
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1,keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1,keepdims=True)
        self.output = probabilities
