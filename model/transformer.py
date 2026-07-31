import torch
import math
import torch.nn as nn

class SelfAttention(nn.Module):

    def __init__(self, embed_dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1,2)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1,2)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1,2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(attention_weights, V)

        out = out.transpose(1,2).contiguous().view(batch_size, seq_len, embed_dim)

        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        self.attention = SelfAttention(embed_dim, n_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x


class GrokkingTransformer(nn.Module):
    def __init__(self, vocab_size, seq_len, embed_dim=128, n_heads=4, n_layers=2, p=97):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(seq_len, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads) for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(embed_dim)
        self.output_layer = nn.Linear(embed_dim, p)

    def forward(self, x):
        batch_size, seq_len = x.shape

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.token_embedding(x) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)

        last_token_output = x[:,-1,:]
        logits = self.output_layer(last_token_output)
        return logits


if __name__ == "__main__":
    p = 5
    vocab_size = p+2
    seq_len = 4

    model = GrokkingTransformer(vocab_size=vocab_size, seq_len=seq_len,
                                 embed_dim=32, n_heads=4, n_layers=2, p=p)

    fake_input = torch.tensor([
        [3, 5, 4, 6],
        [0, 5, 4, 6],
        [2, 5, 0, 6],
    ])

    output = model(fake_input)
    print("Output shape:", output.shape)
    print("Predicted answers:", output.argmax(dim=-1).tolist())