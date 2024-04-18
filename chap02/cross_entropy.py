import numpy as np

class Softmax:
    def __init__(self):
        pass

    def forward(self, input_data, axis=-1):
        # Subtracting the max to prevent overflow
        exps = np.exp(input_data - np.max(input_data, axis=axis, keepdims=True))
        output = exps / np.sum(exps, axis=axis, keepdims=True)
        return output
    

class CrossEntropyLoss_with_Softmax:
    def __init__(self):
        self.softmax = Softmax()

    def forward(self, x, y):
        self.y = y
        self.y_hat = self.softmax.forward(x)
        return -np.sum(np.log(self.y_hat[np.arange(self.y_hat.shape[0]), y] + 1e-8)) / self.y_hat.shape[0]

    ## for one-hot vector
    # def backward(self):
    #     return (self.y_hat - self.y) / self.y_hat.shape[0]
    
    ## otherwise
    def backward(self):
        batch_size = self.y.shape[0]
        dx = self.y_hat.copy()
        dx[np.arange(batch_size), self.y] -= 1
        dx = dx / batch_size
        return dx