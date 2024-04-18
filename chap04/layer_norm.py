import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm_np:
    """
    Numpy implementation of Layer Normalization.
    """
    def __init__(self, epsilon=1e-5):
        self.gamma = None
        self.beta = None
        self.epsilon = epsilon

    def forward(self, x):
        if self.gamma is None or self.beta is None:
            self.gamma = np.ones(x.shape[-1])
            self.beta = np.zeros(x.shape[-1])
        
        mu = np.mean(x, axis=-1, keepdims=True)
        sigma_sq = np.var(x, axis=-1, keepdims=True)
        x_hat = (x - mu) / np.sqrt(sigma_sq + self.epsilon)
        out = self.gamma * x_hat + self.beta
        return out


class LayerNorm(nn.Module):
    """
    PyTorch implementation of Layer Normalization.
    """
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


## check if the two implementations are equivalent
if __name__ == "__main__":
    np.random.seed(0)
    bs = 2
    seq_len = 3
    ndim = 4
    x = np.random.randn(bs, seq_len, ndim)
    ln = LayerNorm_np()
    out_np = ln.forward(x)
    out_np = out_np.astype(np.float32)
    print(out_np)

    ln = LayerNorm(4, bias=True)
    x = torch.tensor(x, dtype=torch.float32)
    out_torch = ln(x)
    print(out_torch)
    assert np.allclose(out_np, out_torch.detach().numpy()), "The outputs are not close."
    assert torch.allclose(torch.tensor(out_np), out_torch), "The outputs are not close."