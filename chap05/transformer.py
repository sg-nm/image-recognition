from typing import Optional
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


@dataclass
class TransformerArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    vocab_size: int = 2000


class Attention(nn.Module):
    def __init__(self, args: TransformerArgs):
        super().__init__()
        self.args = args
        assert args.dim % args.n_heads == 0
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

    def forward(self, x, mask):
        bs, seqlen, _ = x.shape
        q, k, v = self.wq(x), self.wk(x), self.wv(x)

        q = q.view(bs, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bs, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bs, seqlen, self.n_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            attn = attn + mask
        attn = F.softmax(attn, dim=-1)

        x = torch.matmul(attn, v)  # (bs, n_heads, slen, head_dim)
        x = x.transpose(1, 2).contiguous().view(bs, seqlen, -1)
        return self.wo(x)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, args: TransformerArgs):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = FeedForward(dim=args.dim, hidden_dim=4 * args.dim)
        self.attention_norm = nn.LayerNorm(args.dim)
        self.ffn_norm = nn.LayerNorm(args.dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]):
        h = x + self.attention.forward(self.attention_norm(x), mask)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class SimpleTransformer(nn.Module):
    def __init__(self, params: TransformerArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        
        self.layers = torch.nn.ModuleList()
        for _ in range(params.n_layers):
            self.layers.append(TransformerBlock(params))

        self.norm = nn.LayerNorm(params.dim)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

    def forward(self, embedding: torch.Tensor, offset: int=0):
        """
        embedding: [B, L, D]
        embedding has the image tokens and text tokens concatenated.
        offset means the number of image tokens.
        In this function, apply the standard self-attention on image tokens but causal self-attention on text tokens.
        """
        B, seqlen, D = embedding.shape
        
        # causal mask generation for text tokens
        mask = None
        if seqlen > 1:
            bs_mask = 1
            mask = torch.full((bs_mask, 1, seqlen, seqlen), float("-inf"), device=embedding.device)
            mask = torch.triu(mask, diagonal=1).type_as(embedding)
            # compute attention across all image token embeddings
            if offset != 0:
                ii = 0
                ij = 0 + offset
                mask[:, :, ii:ij, ii:ij] = 0.0

        # apply transformer layers: [B, seqlen, D] -> [B, seqlen, D]
        for layer in self.layers:
            embedding = layer(embedding, mask)

        embedding = self.norm(embedding)
        output = self.output(embedding) # [B, seqlen, vocab_size]
        return output