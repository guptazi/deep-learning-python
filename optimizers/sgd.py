# Stochastic Gradient Descent optimizer
class SGD_optimizer:
    def __init__(self, learning_rate=1.0, decay_rate=0.0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.iterations_count = 0

    # Call once before any parameter updates
    def pre_update_parameters(self):
        if self.decay_rate:
            self.current_learning_rate = self.learning_rate * \
                (1.0 / (1.0 + self.decay_rate * self.iterations_count))

    # Update parameters
    def update_parameters(self, layer):
        layer.weights += -self.current_learning_rate * layer.dweights
        layer.biases += -self.current_learning_rate * layer.dbiases

    # Call once after any parameter updates
    def post_update_parameters(self):
        self.iterations_count += 1
