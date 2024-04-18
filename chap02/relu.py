import numpy as np

class ReLU:
    def __init__(self):
        pass

    def forward(self, input_data):
        self.input = input_data
        return np.maximum(0, input_data)

    def backward(self, d_output):
        grad_input = d_output.copy()
        grad_input[self.input < 0] = 0
        return grad_input