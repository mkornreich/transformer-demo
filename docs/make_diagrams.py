#!/usr/bin/env python3
"""
Generate the SVG architecture diagrams used in the README.

Run:  python3 docs/make_diagrams.py
Writes three self-contained SVGs into docs/:
  - transformer-architecture.svg   (Figure 1: encoder-decoder stack)
  - attention.svg                  (Figure 2: scaled dot-product + multi-head)
  - demo-data-flow.svg             (this repo's reverse-a-sequence demo)

The diagrams use an explicit light background so they render legibly on both
light and dark GitHub themes. No external assets or fonts are required.
"""

import os

FONT = "'Segoe UI', Roboto, Helvetica, Arial, sans-serif"

# palette (approximating the paper's pastel scheme)
C = {
    "attn":   ("#ffe1bf", "#e0a45c"),   # multi-head attention (peach)
    "norm":   ("#fff3bf", "#dcc44e"),   # add & norm (yellow)
    "ffn":    ("#cfe6ff", "#6ba6dd"),   # feed forward (blue)
    "embed":  ("#f8d4e8", "#db8fbd"),   # embedding (pink)
    "linear": ("#dfe6f2", "#98b0d3"),   # linear (blue-gray)
    "soft":   ("#d1edd0", "#77bd77"),   # softmax (green)
    "matmul": ("#e7defb", "#a488e0"),   # matmul (purple)
    "scale":  ("#fde3ea", "#e79ab0"),   # scale (rose)
    "mask":   ("#f0f1f4", "#b7bccb"),   # mask (grey, dashed)
    "io":     ("#eef0f6", "#c3c8d6"),   # input/output token boxes
    "cont":   ("#f5f6fa", "#cfd3e0"),   # encoder/decoder container
}
INK = "#1b1b24"
ARROW = "#565a6e"


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def rrect(x, y, w, h, fill, stroke, rx=7, sw=1.6, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def box(cx, cy, w, h, lines, key, fs=12.5, dash=None, fw="500"):
    if isinstance(lines, str):
        lines = [lines]
    fill, stroke = C[key]
    s = [rrect(cx - w / 2, cy - h / 2, w, h, fill, stroke, dash=dash)]
    lh = fs + 2
    y0 = cy - (len(lines) - 1) * lh / 2
    for i, ln in enumerate(lines):
        s.append(f'<text x="{cx:.1f}" y="{y0 + i * lh:.1f}" text-anchor="middle" '
                 f'dominant-baseline="central" font-size="{fs}" font-weight="{fw}" '
                 f'fill="{INK}">{esc(ln)}</text>')
    return "".join(s)


def txt(x, y, s, fs=12, anchor="middle", col=INK, fw="400",
        italic=False, baseline="alphabetic"):
    st = ' font-style="italic"' if italic else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
            f'dominant-baseline="{baseline}" font-size="{fs}" font-weight="{fw}" '
            f'fill="{col}"{st}>{esc(s)}</text>')


def line(x1, y1, x2, y2, col=ARROW, sw=1.6, arrow=True, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    m = ' marker-end="url(#arrow)"' if arrow else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{col}" stroke-width="{sw}"{d}{m}/>')


def poly(points, col=ARROW, sw=1.6, arrow=True, dash=None):
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    d = f' stroke-dasharray="{dash}"' if dash else ""
    m = ' marker-end="url(#arrow)"' if arrow else ""
    return (f'<polyline points="{pts}" fill="none" stroke="{col}" '
            f'stroke-width="{sw}"{d}{m}/>')


def oplus(cx, cy, r=13):
    """circled-plus node used for 'embedding + positional encoding'."""
    return (f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#ffffff" '
            f'stroke="{ARROW}" stroke-width="1.6"/>'
            f'<line x1="{cx-r+4}" y1="{cy}" x2="{cx+r-4}" y2="{cy}" '
            f'stroke="{ARROW}" stroke-width="1.6"/>'
            f'<line x1="{cx}" y1="{cy-r+4}" x2="{cx}" y2="{cy+r-4}" '
            f'stroke="{ARROW}" stroke-width="1.6"/>')


def sine(cx, cy, w=34, amp=5, col="#9b8ccc"):
    x0 = cx - w / 2
    q = w / 4
    d = (f"M {x0:.1f},{cy} q {q:.1f},{-amp} {2*q:.1f},0 "
         f"q {q:.1f},{amp} {2*q:.1f},0")
    return f'<path d="{d}" fill="none" stroke="{col}" stroke-width="1.8"/>'


def wrap(w, h, body, title=None):
    defs = (f'<defs><marker id="arrow" viewBox="0 0 10 10" refX="8.5" refY="5" '
            f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{ARROW}"/></marker></defs>')
    bg = rrect(1, 1, w - 2, h - 2, "#ffffff", "#e7e9f1", rx=14, sw=1.4)
    t = txt(w / 2, 30, title, fs=16, fw="600") if title else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="{w}" height="{h}" font-family="{FONT}">'
            f'{defs}{bg}<g font-family="{FONT}">{t}{body}</g></svg>')


# ---------------------------------------------------------------------------
# Figure 1 : the full encoder-decoder architecture
# ---------------------------------------------------------------------------
def architecture():
    W, H = 780, 700
    ex, dx = 220, 560          # encoder / decoder column centres
    bw = 180                   # box width
    b = []

    # containers
    b.append(rrect(ex - bw/2 - 16, 301, bw + 32, 214, *C["cont"], rx=12,
                   sw=1.4, dash="6 5"))
    b.append(rrect(dx - bw/2 - 16, 205, bw + 32, 310, *C["cont"], rx=12,
                   sw=1.4, dash="6 5"))
    b.append(txt(ex - bw/2 - 4, 296, "N×", fs=12.5, anchor="end",
                 col="#6b7086", fw="600"))
    b.append(txt(dx - bw/2 - 4, 200, "N×", fs=12.5, anchor="end",
                 col="#6b7086", fw="600"))

    # --- encoder boxes (bottom -> top) ---
    b.append(box(ex, 472, bw, 42, "Multi-Head Attention", "attn"))
    b.append(box(ex, 425, bw, 28, "Add & Norm", "norm", fs=12))
    b.append(box(ex, 376, bw, 42, "Feed Forward", "ffn"))
    b.append(box(ex, 329, bw, 28, "Add & Norm", "norm", fs=12))

    # --- decoder boxes (bottom -> top) ---
    b.append(box(dx, 472, bw, 42, ["Masked Multi-Head", "Attention"], "attn"))
    b.append(box(dx, 425, bw, 28, "Add & Norm", "norm", fs=12))
    b.append(box(dx, 376, bw, 42, ["Multi-Head", "Attention"], "attn"))
    b.append(box(dx, 329, bw, 28, "Add & Norm", "norm", fs=12))
    b.append(box(dx, 280, bw, 42, "Feed Forward", "ffn"))
    b.append(box(dx, 233, bw, 28, "Add & Norm", "norm", fs=12))

    # --- top of decoder: linear + softmax + output ---
    b.append(box(dx, 155, bw, 30, "Linear", "linear", fs=12.5))
    b.append(box(dx, 105, bw, 30, "Softmax", "soft", fs=12.5))
    b.append(txt(dx, 66, "Output Probabilities", fs=12.5, fw="500"))

    # --- embeddings + positional encoding ---
    for cx, lbl in ((ex, "Input Embedding"), (dx, "Output Embedding")):
        b.append(oplus(cx, 550))
        b.append(box(cx, 600, bw, 34, lbl, "embed", fs=12.5))
    b.append(sine(ex - 108, 550)); b.append(sine(dx + 108, 550))
    b.append(txt(ex - 108, 517, "Positional", fs=9.5, col="#6b7086"))
    b.append(txt(ex - 108, 528, "Encoding", fs=9.5, col="#6b7086"))
    b.append(txt(dx + 108, 517, "Positional", fs=9.5, col="#6b7086"))
    b.append(txt(dx + 108, 528, "Encoding", fs=9.5, col="#6b7086"))

    # --- bottom io labels ---
    b.append(txt(ex, 651, "Inputs", fs=12.5, fw="500"))
    b.append(txt(dx, 651, "Outputs", fs=12.5, fw="500"))
    b.append(txt(dx, 665, "(shifted right)", fs=10, col="#6b7086"))

    # --- vertical flow arrows (encoder) ---
    for (y1, y2) in [(641, 619), (583, 565), (537, 495), (451, 441),
                     (411, 399), (355, 345)]:
        b.append(line(ex, y1, ex, y2))
    b.append(line(ex, 315, ex, 303))                      # -> container top
    # --- vertical flow arrows (decoder) ---
    for (y1, y2) in [(641, 619), (583, 565), (537, 495), (451, 441),
                     (411, 399), (355, 343), (315, 303), (259, 249),
                     (219, 207)]:
        b.append(line(dx, y1, dx, y2))
    b.append(line(dx, 205, dx, 172))                      # -> Linear
    b.append(line(dx, 140, dx, 122))                      # -> Softmax
    b.append(line(dx, 90, dx, 80))                        # -> Output Prob

    # --- encoder output feeds decoder cross-attention (K, V) ---
    b.append(poly([(ex, 301), (ex, 289), (415, 289), (415, 376), (468, 376)]))
    b.append(txt(360, 281, "encoder output  (K, V)", fs=10,
                 col="#6b7086"))

    # --- footnote ---
    b.append(txt(W/2, 688,
                 "inputs = token embedding × sqrt(d_model)  +  sinusoidal "
                 "positional encoding  (§3.5)     •     "
                 "each sub-layer = LayerNorm(x + Sublayer(x))  (§3.1)",
                 fs=10, col="#6b7086"))

    return wrap(W, H, "".join(b), "The Transformer — Encoder-Decoder Architecture")


# ---------------------------------------------------------------------------
# Figure 2 : scaled dot-product attention (left) + multi-head attention (right)
# ---------------------------------------------------------------------------
def attention():
    W, H = 820, 430
    b = []

    # ---------- left: scaled dot-product attention ----------
    lx = 205
    lw = 128
    b.append(txt(lx, 34, "Scaled Dot-Product Attention", fs=14, fw="600"))
    b.append(box(lx, 108, 96, 30, "MatMul", "matmul", fs=12))
    b.append(box(lx, 158, 96, 30, "SoftMax", "soft", fs=12))
    b.append(box(lx, 208, 96, 30, "Mask (opt.)", "mask", fs=11.5, dash="5 4"))
    b.append(box(lx, 258, 96, 30, "Scale", "scale", fs=12))
    b.append(box(lx, 308, 96, 30, "MatMul", "matmul", fs=12))
    # inputs
    b.append(txt(lx - 34, 372, "Q", fs=13, fw="600"))
    b.append(txt(lx, 372, "K", fs=13, fw="600"))
    b.append(txt(lx + 60, 372, "V", fs=13, fw="600"))
    # arrows up the main column
    b.append(line(lx, 293, lx, 273))      # matmul(QK) -> scale
    b.append(line(lx, 243, lx, 223))      # scale -> mask
    b.append(line(lx, 193, lx, 173))      # mask -> softmax
    b.append(line(lx, 143, lx, 123))      # softmax -> matmul(.V)
    b.append(line(lx, 93, lx, 72))        # matmul -> output
    b.append(txt(lx, 62, "output", fs=10.5, col="#6b7086"))
    # Q, K into bottom matmul
    b.append(poly([(lx - 34, 362), (lx - 34, 323), (lx - 20, 323)]))
    b.append(poly([(lx, 362), (lx, 323)]))
    # V up the side into top matmul
    b.append(poly([(lx + 60, 362), (lx + 60, 108), (lx + 48, 108)]))

    # ---------- right: multi-head attention ----------
    rx = 570
    b.append(txt(rx, 34, "Multi-Head Attention", fs=14, fw="600"))
    b.append(box(rx, 100, 108, 30, "Linear", "linear", fs=12))
    b.append(box(rx, 152, 118, 30, "Concat", "io", fs=12))
    # stacked "scaled dot-product attention" (h heads)
    b.append(rrect(rx - 95 + 10, 210 - 25 + 10, 190, 50, *C["matmul"], rx=8,
                   sw=1.4))
    b.append(rrect(rx - 95 + 5, 210 - 25 + 5, 190, 50, *C["matmul"], rx=8,
                   sw=1.4))
    b.append(box(rx, 214, 190, 50,
                 ["Scaled Dot-Product", "Attention"], "matmul", fs=12))
    b.append(txt(rx + 118, 200, "h", fs=12.5, italic=True, col="#6b7086"))
    # three linear projections
    b.append(box(rx - 60, 300, 56, 28, "Linear", "linear", fs=11))
    b.append(box(rx, 300, 56, 28, "Linear", "linear", fs=11))
    b.append(box(rx + 60, 300, 56, 28, "Linear", "linear", fs=11))
    # inputs V K Q
    b.append(txt(rx - 60, 372, "V", fs=13, fw="600"))
    b.append(txt(rx, 372, "K", fs=13, fw="600"))
    b.append(txt(rx + 60, 372, "Q", fs=13, fw="600"))
    for dxi in (-60, 0, 60):
        b.append(line(rx + dxi, 362, rx + dxi, 314))      # input -> linear
        b.append(line(rx + dxi, 286, rx + dxi, 240))      # linear -> SDPA
    b.append(line(rx, 189, rx, 168))     # SDPA -> concat
    b.append(line(rx, 137, rx, 116))     # concat -> linear
    b.append(line(rx, 85, rx, 66))       # linear -> output
    b.append(txt(rx, 56, "output", fs=10.5, col="#6b7086"))

    # divider + caption
    b.append(line(400, 60, 400, 370, col="#e4e6ee", sw=1.2, arrow=False))
    b.append(txt(W/2, 406,
                 "Attention(Q,K,V) = softmax(QKᵀ / sqrt(dₖ)) V   "
                 "(Eq. 1)      •      dₖ = d_v = d_model / h   "
                 "(here h = 4, dₖ = 16)",
                 fs=10.5, col="#6b7086"))
    return wrap(W, H, "".join(b))


# ---------------------------------------------------------------------------
# This repo's demo: learn to reverse a digit sequence
# ---------------------------------------------------------------------------
def demo_flow():
    W, H = 940, 300
    cy = 150
    b = []
    b.append(txt(W/2, 32, "This demo — learning to reverse a sequence",
                 fs=15, fw="600"))

    stages = [
        (80,  110, ["Input", "3 1 4 1 5 9 2 6"], "io"),
        (238, 128, ["Embedding", "× sqrt(d_model)", "+ positional enc."], "embed"),
        (392, 110, ["Encoder", "× N=2"], "attn"),
        (562, 150, ["Decoder × N=2", "masked self-attn", "+ cross-attn"], "attn"),
        (722, 112, ["Linear", "+ Softmax"], "soft"),
        (862, 110, ["Output", "6 2 9 5 1 4 1 3"], "io"),
    ]
    for cx, w, lines, key in stages:
        b.append(box(cx, cy, w, 74, lines, key, fs=11.5))

    # arrows between stages
    edges = [(135, 178), (302, 337), (447, 492), (632, 667), (778, 807)]
    for x1, x2 in edges:
        b.append(line(x1, cy, x2, cy))
    b.append(txt((447 + 492) / 2, cy - 12, "memory", fs=9.5, col="#6b7086"))
    b.append(txt((447 + 492) / 2, cy - 2, "(K,V)", fs=9.5, col="#6b7086"))

    # autoregressive feedback loop: softmax output -> decoder input
    b.append(poly([(722, 187), (722, 240), (562, 240), (562, 187)],
                  dash="5 4"))
    b.append(txt(642, 254, "greedy decode: feed prediction back (teacher-free)",
                 fs=9.5, col="#6b7086"))

    # config chip
    b.append(rrect(W/2 - 210, 272, 420, 22, "#f5f6fa", "#cfd3e0", rx=11, sw=1.2))
    b.append(txt(W/2, 283,
                 "d_model = 64   •   h = 4   •   N = 2   •   "
                 "d_ff = 256   •   ~235K params   •   100% held-out "
                 "accuracy in ~25s (CPU)",
                 fs=10, col="#4a4f63", baseline="central"))
    return wrap(W, H, "".join(b))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out = {
        "transformer-architecture.svg": architecture(),
        "attention.svg": attention(),
        "demo-data-flow.svg": demo_flow(),
    }
    for name, svg in out.items():
        path = os.path.join(here, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(svg)
        print(f"wrote {path}  ({len(svg):,} bytes)")


if __name__ == "__main__":
    main()
