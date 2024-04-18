from fc import FullyConnectedLayer
from relu import ReLU

# Three-layer MLP
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.layers = [
            FullyConnectedLayer(input_dim, hidden_dim),
            ReLU(),
            FullyConnectedLayer(hidden_dim, output_dim),
        ]

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    # compute the gradient of each layer and backprop it
    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output

    # update the weights of each layer
    def update(self, learning_rate):
        for layer in self.layers:
            if isinstance(layer, FullyConnectedLayer):
                layer.update(learning_rate)