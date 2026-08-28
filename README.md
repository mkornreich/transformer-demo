# Transformer Demo — *Attention Is All You Need*

A single-file, from-scratch implementation of the Transformer from
[Vaswani et al., 2017 — *Attention Is All You Need*](https://arxiv.org/abs/1706.03762),
trained on a toy sequence-reversal task so you can watch it learn.

```bash
python3 transformer_demo.py
```

Trains an encoder–decoder Transformer to reverse a sequence of digits
(`3 1 4 1 5 → 5 1 4 1 3`), then greedy-decodes held-out examples.
On CPU it reaches **100% token accuracy in ~25 seconds**.

## What's implemented from the paper

Only `nn.Linear` / `nn.Embedding` / `nn.LayerNorm` / `nn.Dropout` and the Adam
optimizer are borrowed — attention, multi-head, and positional encoding are all
written by hand. No `nn.Transformer` / `nn.MultiheadAttention`.

| Paper | Code |
|---|---|
| Scaled Dot-Product Attention, Eq. 1 — `softmax(QKᵀ/√dₖ)V` | `scaled_dot_product_attention` |
| Multi-Head Attention, §3.2.2 — split into *h* heads, concat, `Wᴼ` | `MultiHeadAttention` |
| Sinusoidal Positional Encoding, §3.5 | `PositionalEncoding` |
| Position-wise FFN, Eq. 2 — `max(0, xW₁+b₁)W₂+b₂`, `d_ff = 4·d_model` | `FeedForward` |
| Residual + LayerNorm sublayers, §3.1 | `EncoderLayer` / `DecoderLayer` |
| Causal-masked decoder self-attn + encoder–decoder cross-attn, §3.2.3 | `DecoderLayer`, `causal_mask` |
| `√d_model` embedding scaling, §3.4 | `Transformer._emb` |

The script also prints a one-time tensor-shape trace of the forward pass and an
ASCII heatmap of the decoder's cross-attention — which comes out as a clean
anti-diagonal, since reversal means output position *t* attends to source
position *(n−t)*.

## Config

`N=2` layers, `d_model=64`, `h=4` heads, `d_ff=256` — ~235K parameters, small
enough to train quickly on CPU.

## Requirements

- Python 3
- `torch`
- `numpy`

## License

The paper itself is not included here; see the
[arXiv page](https://arxiv.org/abs/1706.03762). This demo code is provided as-is
for educational use.
