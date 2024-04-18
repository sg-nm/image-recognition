## train mnist

import numpy as np

from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from mlp import MLP
from cross_entropy import CrossEntropyLoss_with_Softmax

# Load the MNIST dataset
mnist_data = MNIST('./mnist', train=True, download=True, transform=transforms.ToTensor())
data_loader = DataLoader(mnist_data, batch_size=256, shuffle=True)

# Training hyper-parameters
num_epochs = 50
learning_rate = 0.05

# Create the MLP
mlp = MLP(784, 128, 10)

# Create the objective function
criterion = CrossEntropyLoss_with_Softmax()

# Training loop
for epoch in range(num_epochs):
    loss = 0.0
    accuracy = 0.0
    # X_mini: (batch_size, 1, 28, 28)
    # y_mini: (batch_size,)
    for i, (X_mini, y_mini) in enumerate(data_loader):
        # convert the tensor to numpy array
        X_mini = X_mini.view(X_mini.size(0), -1).numpy()
        y_mini = y_mini.numpy()
        y_pred = mlp.forward(X_mini)
        loss += criterion.forward(y_pred, y_mini)

        # Back-propagation
        grad_output = criterion.backward()
        mlp.backward(grad_output)

        # update the weight parameters
        mlp.update(learning_rate)

        # Calculate accuracy
        y_pred = np.argmax(y_pred, axis=1)
        # y_true = np.argmax(y_mini, axis=1)
        accuracy += np.mean(y_mini == y_pred) * 100
    
    loss /= len(data_loader)
    accuracy /= len(data_loader)
        
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss}, Accuracy: {accuracy}%')

