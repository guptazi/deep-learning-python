class Dropout_layer:
  # Init
  def __init__(self, rate):
    # of 0.1 we need success rate of 0.9
    self.rate = 1 - rate
  # Forward pass
  def forward(self, inputs):
    # Save input values
    self.inputs = inputs
    # Generate and save scaled mask
    self.binary_mask = np.random.binomial(1, self.rate,
    size=inputs.shape) / self.rate
    # Apply mask to output values
    self.output = inputs * self.binary_mask
  # Backward pass
  def backward(self, dvalues):
    # Gradient on values
    self.dinputs = dvalues * self.binary_mask
