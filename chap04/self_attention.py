import math
from torch.nn import Linear
from torch.nn import functional as F

class SelfAttention:
    """
    img:       (B, L, embed_dim)
    B:         mini-batch size
    L:         number of patches
    embed_dim: embedding size
    n_heads:   number of heads
    """
    def __init__(self, embed_dim, n_heads):
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        self.fc_q = Linear(embed_dim, embed_dim)
        self.fc_k = Linear(embed_dim, embed_dim)
        self.fc_v = Linear(embed_dim, embed_dim)
        self.fc_o = Linear(embed_dim, embed_dim)

    def forward(self, img):
        B, L, D = img.shape
        assert D == self.embed_dim, f"Expected {self.embed_dim} but got {D}"

        # apply linear layers: [B, L, D]
        q = self.fc_q(img)
        k = self.fc_k(img)
        v = self.fc_v(img)

        # multi-heads: [B, n_heads, L, head_dim]
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # [B, n_heads, L, head_dim] @ [B, n_heads, head_dim, L] -> [B, n_heads, L, L]
        attention = q @ k.transpose(-2, -1)
        attention = attention / math.sqrt(self.head_dim)
        attention = F.softmax(attention, dim=-1)

        # [B, n_heads, L, L] @ [B, n_heads, L, head_dim] -> [B, n_heads, L, head_dim]
        out = attention @ v
        
        # [B, L, n_heads, head_dim]
        out = out.transpose(1, 2).contiguous()
        # [B, L, embed_dim]
        out = out.view(B, L, D)
        
        out = self.fc_o(out)
        return out



if __name__ == "__main__":
    import torch
    img_seq = torch.randn(2, 10, 512)
    sa = SelfAttention(512, 8)
    out = sa.forward(img_seq)
    print(out.shape)  # torch.Size([2, 10, 512])