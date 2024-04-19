import torch
from torch import nn
from transformer import SimpleTransformer, TransformerArgs


class SimplePix2Seq:
    """
    img:  [batch_size, C, H, W]
    text: [batch_size, seq_len]
    """
    def __init__(self, dim, num_heads, num_layers, vocab_size, enc_seq_len, seq_len):
        # encoder (any vision model that outputs a sequence of features can be used here)
        self.encoder = VisionTransformer()
        self.enc_pos_emb = nn.Embedding(enc_seq_len, dim)

        # decoder
        self.embedding = nn.Embedding(vocab_size, dim)
        self.dec_pos_emb = nn.Embedding(seq_len, dim)
        model_args = TransformerArgs(dim=dim, n_layers=num_layers, n_heads=num_heads, vocab_size=vocab_size)
        self.decoder = SimpleTransformer(model_args)

        # head
        self.head = nn.Linear(dim, vocab_size)
    
    def forward(self, img, text):
        # enc_out: [batch_size, enc_seq_len, dim]
        enc_out = self.encoder(img)
        B, num_img_tokens, D = enc_out.shape
        pos = torch.arange(0, num_img_tokens, dtype=torch.long, device=enc_out.device)
        enc_pos_emb = self.enc_pos_emb(pos)
        enc_out = enc_out + enc_pos_emb

        # dec_out: [batch_size, seq_len, dim]
        B, seq_len = text.shape
        text_emb = self.embedding(text)
        dec_pos_emb = self.dec_pos_emb(seq_len)
        text_emb = text_emb + dec_pos_emb
        
        # concat enc_out and text_emb: [batch_size, enc_seq_len + seq_len, dim]
        input_embeddings = torch.cat([enc_out, text_emb], dim=1)
        dec_out = self.decoder(input_embeddings, offset=num_img_tokens)

        # head : [batch_size, seq_len, vocab_size]
        y = self.head(dec_out)
        return y