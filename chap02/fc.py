import numpy as np

class FullyConnectedLayer:
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.randn(input_dim, output_dim)
        self.bias = np.zeros(output_dim)

    def forward(self, X):
        self.X = X
        return np.dot(X, self.weights) + self.bias

    # compute the gradient w.r.t. the weight parameters
    def backward(self, grad_output):
        grad_input = np.dot(grad_output, self.weights.T)
        self.grad_weights = np.dot(self.X.T, grad_output)
        self.grad_bias = np.sum(grad_output, axis=0)
        return grad_input

    # update the weight parameters
    def update(self, learning_rate):
        self.weights -= learning_rate * self.grad_weights
        self.bias -= learning_rate * self.grad_bias