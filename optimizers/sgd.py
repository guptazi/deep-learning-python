class Optimizer_SGD:
  def __init__(self, learning_rate=1.0):
    self.learning_rate = learning_rate
    # Update parameters
  def params_update(self, layer):
    layer.weights += -self.learning_rate * layer.dweights
    layer.biases += -self.learning_rate * layer.dbiases
