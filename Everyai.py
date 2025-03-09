import numpy as np

# ReLU and Sigmoid Activation Functions
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Neural Network Architecture
input_neurons = 9000
hidden_layers = 800  # Number of hidden layers
hidden_neurons = 10000
output_neurons = 9000

# Bias values
input_bias = 5
hidden_bias = 4
output_bias = 5

# Weight initialization (as per your original code)
input_weights = np.random.randn(input_neurons, hidden_neurons) * 0.01

# Initialize hidden layer weights for multiple hidden layers
hidden_weights = []
hidden_weights.append(np.random.randn(hidden_neurons, hidden_neurons) * 0.01)

output_weights = np.random.randn(hidden_neurons, output_neurons) * 0.01

# Forward Pass
def forward_pass(X):
    # Input to first hidden layer
    z1 = np.dot(X, input_weights) + input_bias
    a1 = relu(z1)  # ReLU activation

    # Hidden layers
    hidden_output = a1
    for weight in hidden_weights:
        z = np.dot(hidden_output, weight) + hidden_bias
        hidden_output = relu(z)  # Apply ReLU activation

    # Output layer (apply sigmoid)
    z_out = np.dot(hidden_output, output_weights) + output_bias
    a_out = sigmoid(z_out)  # Sigmoid activation for the output layer

    return a_out

# Example input data (1 sample, 9000 features)
X = np.random.randn(1, input_neurons)

# Perform forward pass
output = forward_pass(X)
print("Output after forward pass:", output)
