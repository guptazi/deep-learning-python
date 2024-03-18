class Optimizer_RMSprop:
    # Initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay_rate=0.0, epsilon_val=1e-7, rho_val=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.iterations_count = 0
        self.epsilon = epsilon_val
        self.rho = rho_val
    
    # Call once before any parameter updates
    def pre_update_parameters(self):
        if self.decay_rate:
            self.current_learning_rate = self.learning_rate * \
                (1.0 / (1.0 + self.decay_rate * self.iterations_count))
    
    # Update parameters
    def update_parameters(self, layer):
        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        # Update cache with squared current gradients
        layer.weight_cache = self.rho * layer.weight_cache + \
            (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + \
            (1 - self.rho) * layer.dbiases**2
        
        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * \
            layer.dweights / \
            (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
            layer.dbiases / \
            (np.sqrt(layer.bias_cache) + self.epsilon)
    
    # Call once after any parameter updates
    def post_update_parameters(self):
        self.iterations_count += 1
