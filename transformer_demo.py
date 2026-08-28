"""
Attention Is All You Need (Vaswani et al., 2017) -- minimal, self-contained demo.

Task (a clean "watch it learn" seq2seq problem): reverse a sequence of digits.
    input  3 1 4 1 5 9 2 6   ->   output  6 2 9 5 1 4 1 3

Everything essential is implemented FROM SCRATCH: scaled dot-product attention
(Eq. 1), multi-head attention (Sec. 3.2.2), sinusoidal positional encoding
(Sec. 3.5), the position-wise FFN (Eq. 2), and the full encoder/decoder stacks
with residual + LayerNorm sublayers (Sec. 3.1). Only nn.Linear / nn.Embedding /
nn.LayerNorm / nn.Dropout and a torch optimizer are borrowed; no nn.Transformer
or nn.MultiheadAttention. torch + numpy + stdlib only. Runs on CPU in ~25s.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)
torch.set_num_threads(4)   # avoid CPU oversubscription on a shared box

# ---------------------------------------------------------------------------
# Config (small, so the whole thing trains on CPU in well under a minute)
# ---------------------------------------------------------------------------
D_MODEL = 64        # embedding / model dimension
N_HEADS = 4         # attention heads;  d_k = d_v = D_MODEL / N_HEADS = 16
N_LAYERS = 2        # identical layers in each of encoder and decoder
D_FF = 4 * D_MODEL  # inner FFN dimension (paper uses 4 * d_model)
DROPOUT = 0.0       # fresh random batches each step -> nothing to overfit to

SEQ_LEN = 8         # length of the integer sequences to reverse
VOCAB = 12          # symbols 0..9 plus BOS(10) and EOS(11)
BOS, EOS = 10, 11


# ---------------------------------------------------------------------------
# Scaled Dot-Product Attention -- Eq. (1):
#     Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V
# The 1/sqrt(d_k) scaling keeps dot products small; softmax stays in a
# high-gradient region. Also returns the weights, so we can visualize them.
# ---------------------------------------------------------------------------
def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:                       # mask==0 -> forbidden position
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, v), attn


# ---------------------------------------------------------------------------
# Multi-Head Attention -- Sec. 3.2.2:
#     MultiHead(Q,K,V) = Concat(head_1..head_h) W^O
#     head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V),  d_k = d_v = d_model/h
# The per-head projections are packed into single Linear layers, then reshaped.
# ---------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)   # all W_i^Q stacked
        self.w_k = nn.Linear(d_model, d_model)   # all W_i^K stacked
        self.w_v = nn.Linear(d_model, d_model)   # all W_i^V stacked
        self.w_o = nn.Linear(d_model, d_model)   # output projection W^O
        self.drop = nn.Dropout(dropout)
        self.last_attn = None                    # cached for the heatmap

    def _split(self, x):                         # (B,T,d_model) -> (B,h,T,d_k)
        B, T, _ = x.shape
        return x.view(B, T, self.h, self.d_k).transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        q = self._split(self.w_q(query))
        k = self._split(self.w_k(key))
        v = self._split(self.w_v(value))
        if mask is not None:
            mask = mask.unsqueeze(1)             # broadcast over the head axis
        out, attn = scaled_dot_product_attention(q, k, v, mask)  # (B,h,T,d_k)
        self.last_attn = attn.detach()
        B, _, T, _ = out.shape
        out = out.transpose(1, 2).contiguous().view(B, T, self.h * self.d_k)
        return self.drop(self.w_o(out))          # concat heads, then W^O


# ---------------------------------------------------------------------------
# Position-wise Feed-Forward Network -- Eq. (2):
#     FFN(x) = max(0, x W1 + b1) W2 + b2       (inner dim d_ff = 4 * d_model)
# Applied identically to each position.
# ---------------------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.drop(F.relu(self.w1(x))))


# ---------------------------------------------------------------------------
# Sinusoidal Positional Encoding -- Sec. 3.5:
#     PE(pos,2i)   = sin(pos / 10000^(2i/d_model))
#     PE(pos,2i+1) = cos(pos / 10000^(2i/d_model))
# The model has no recurrence/convolution, so position info is added in here.
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        # 1 / 10000^(2i/d_model), computed in log-space for numerical stability.
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)       # even dims -> sin
        pe[:, 1::2] = torch.cos(pos * div)       # odd  dims -> cos
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


# ---------------------------------------------------------------------------
# One encoder layer: self-attention + FFN, each wrapped as
#     LayerNorm(x + Sublayer(x))                (Sec. 3.1, residual + norm)
# ---------------------------------------------------------------------------
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, src_mask):
        x = self.norm1(x + self.self_attn(x, x, x, src_mask))   # sub-layer 1
        x = self.norm2(x + self.ffn(x))                         # sub-layer 2
        return x


# ---------------------------------------------------------------------------
# One decoder layer: masked self-attention, encoder-decoder (cross) attention,
# then FFN -- each again wrapped as LayerNorm(x + Sublayer(x)).
# ---------------------------------------------------------------------------
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, memory, tgt_mask, src_mask):
        # 1. causally masked self-attention (no attending to future positions)
        x = self.norm1(x + self.self_attn(x, x, x, tgt_mask))
        # 2. cross-attention: queries from decoder, keys/values from encoder
        x = self.norm2(x + self.cross_attn(x, memory, memory, src_mask))
        # 3. position-wise feed-forward
        x = self.norm3(x + self.ffn(x))
        return x


# ---------------------------------------------------------------------------
# Full encoder-decoder Transformer.  Token embeddings are scaled by
# sqrt(d_model) (Sec. 3.4); a final linear layer produces token logits.
# ---------------------------------------------------------------------------
class Transformer(nn.Module):
    def __init__(self, vocab, d_model, n_heads, n_layers, d_ff, dropout):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab, d_model)
        self.pos = PositionalEncoding(d_model)
        self.drop = nn.Dropout(dropout)
        self.enc = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.dec = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.proj = nn.Linear(d_model, vocab)     # final linear -> logits
        self._traced = False                      # for a one-time shape trace

    def _emb(self, x):
        return self.drop(self.pos(self.embed(x) * math.sqrt(self.d_model)))

    def encode(self, src, src_mask):
        x = self._emb(src)
        for layer in self.enc:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, memory, tgt_mask, src_mask):
        x = self._emb(tgt)
        for layer in self.dec:
            x = layer(x, memory, tgt_mask, src_mask)
        return x

    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encode(src, src_mask)
        out = self.decode(tgt, memory, tgt_mask, src_mask)
        logits = self.proj(out)
        if not self._traced:                      # print shapes once, for readers
            print("  [shape trace] src embedding   :", tuple(self._emb(src).shape))
            print("  [shape trace] encoder memory  :", tuple(memory.shape))
            print("  [shape trace] decoder output  :", tuple(out.shape))
            print("  [shape trace] logits (vocab)  :", tuple(logits.shape), "\n")
            self._traced = True
        return logits


def causal_mask(T):
    # lower-triangular: position t may attend only to <= t  (Sec. 3.2.3)
    return torch.tril(torch.ones(1, T, T)).long()


# ---------------------------------------------------------------------------
# Toy data: reverse a length-SEQ_LEN sequence of digits (0..9).
# Decoder is fed BOS + reversed sequence (teacher forcing); it must predict
# reversed sequence + EOS.
# ---------------------------------------------------------------------------
def make_batch(batch_size):
    src = torch.randint(0, 10, (batch_size, SEQ_LEN))
    rev = torch.flip(src, dims=[1])
    bos = torch.full((batch_size, 1), BOS)
    eos = torch.full((batch_size, 1), EOS)
    tgt_in = torch.cat([bos, rev], dim=1)          # decoder input
    tgt_out = torch.cat([rev, eos], dim=1)         # expected output
    return src, tgt_in, tgt_out


# ---------------------------------------------------------------------------
# A small ASCII heatmap of decoder cross-attention for one example, showing the
# interpretable alignment the paper highlights: output position t attends to the
# source position it copies (reversal -> a clean anti-diagonal).
# ---------------------------------------------------------------------------
def ascii_heatmap(weights, xs, ys):
    chars = " .:-=+*#%@"
    w = weights / (weights.max() + 1e-9)
    print("        " + " ".join(f"{x:>2}" for x in xs) + "   <- source (input)")
    for i, row in enumerate(w):
        cells = " ".join(chars[int(v * (len(chars) - 1) + 0.5)] + " " for v in row)
        print(f"  out {ys[i]:>2}  {cells}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def main():
    model = Transformer(VOCAB, D_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, betas=(0.9, 0.98), eps=1e-9)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Transformer: N={N_LAYERS} d_model={D_MODEL} h={N_HEADS} "
          f"d_ff={D_FF} params={n_params:,}")
    print("Task: reverse a length-%d digit sequence\n" % SEQ_LEN)

    EPOCHS, STEPS, BATCH = 40, 50, 64
    src_mask = torch.ones(1, 1, SEQ_LEN).long()    # no padding -> attend all
    tmask = causal_mask(SEQ_LEN + 1)               # decoder length = SEQ_LEN+1

    model.train()
    for epoch in range(1, EPOCHS + 1):
        total = 0.0
        for _ in range(STEPS):
            src, tgt_in, tgt_out = make_batch(BATCH)
            logits = model(src, tgt_in, src_mask, tmask)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), tgt_out.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"epoch {epoch:2d}/{EPOCHS}   loss {total / STEPS:.4f}")

    # -----------------------------------------------------------------------
    # Greedy autoregressive decoding on held-out examples (Sec. 3.1 inference):
    # start from BOS and feed each predicted token back in.
    # -----------------------------------------------------------------------
    model.eval()
    correct = tokens = 0
    shown = 0
    with torch.no_grad():
        for _ in range(20):
            src, _, tgt_out = make_batch(32)
            memory = model.encode(src, src_mask)
            ys = torch.full((src.size(0), 1), BOS)
            for _ in range(SEQ_LEN):
                m = causal_mask(ys.size(1))
                out = model.decode(ys, memory, m, src_mask)
                nxt = model.proj(out[:, -1]).argmax(-1, keepdim=True)
                ys = torch.cat([ys, nxt], dim=1)
            pred = ys[:, 1:]                         # drop BOS
            gold = tgt_out[:, :SEQ_LEN]              # reversed src (before EOS)
            correct += (pred == gold).sum().item()
            tokens += gold.numel()
            while shown < 5:
                i = shown
                ok = "OK" if torch.equal(pred[i], gold[i]) else "x"
                print(f"  in {src[i].tolist()}  ->  pred {pred[i].tolist()}  [{ok}]")
                shown += 1

    acc = 100.0 * correct / tokens
    print(f"\nHeld-out token accuracy: {acc:.2f}%  ({correct}/{tokens})")

    # -----------------------------------------------------------------------
    # Interpretability: last decoder layer, head 0 cross-attention for one
    # example. Rows = output positions, cols = source positions.
    # -----------------------------------------------------------------------
    with torch.no_grad():
        src, tgt_in, _ = make_batch(1)
        _ = model(src, tgt_in, src_mask, tmask)     # populate last_attn
        attn = model.dec[-1].cross_attn.last_attn[0, 0, :SEQ_LEN]  # (SEQ_LEN, SEQ_LEN)
        print("\nDecoder cross-attention (last layer, head 0) -- output attends to source:")
        ascii_heatmap(attn.numpy(), src[0].tolist(), torch.flip(src, dims=[1])[0].tolist())


if __name__ == "__main__":
    main()
