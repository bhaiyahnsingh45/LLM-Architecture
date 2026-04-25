# Transformer Architecture — From Scratch

A personal study repository for learning the Transformer architecture bottom-up. Notes, derivations, and code will be added here incrementally.

> Reference paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017

---

## Table of Contents

1. [Big Picture](#1-big-picture)
2. [Input Pipeline](#2-input-pipeline)
   - 2.1 [Tokenization](#21-tokenization)
   - 2.2 [Token Embeddings](#22-token-embeddings)
   - 2.3 [Positional Encoding](#23-positional-encoding)
3. [Attention Mechanism](#3-attention-mechanism)
   - 3.1 [Scaled Dot-Product Attention](#31-scaled-dot-product-attention)
   - 3.2 [Multi-Head Attention](#32-multi-head-attention)
   - 3.3 [Self-Attention vs Cross-Attention](#33-self-attention-vs-cross-attention)
   - 3.4 [Causal (Masked) Attention](#34-causal-masked-attention)
4. [Feed-Forward Network](#4-feed-forward-network)
5. [Layer Normalization](#5-layer-normalization)
6. [Residual Connections](#6-residual-connections)
7. [The Encoder](#7-the-encoder)
8. [The Decoder](#8-the-decoder)
9. [Encoder-Decoder Stack](#9-encoder-decoder-stack)
10. [Output Head](#10-output-head)
    - 10.1 [Linear Projection](#101-linear-projection)
    - 10.2 [Softmax & Sampling Strategies](#102-softmax--sampling-strategies)
11. [Training](#11-training)
    - 11.1 [Loss Function (Cross-Entropy)](#111-loss-function-cross-entropy)
    - 11.2 [Label Smoothing](#112-label-smoothing)
    - 11.3 [Optimizer & Learning Rate Schedule](#113-optimizer--learning-rate-schedule)
    - 11.4 [Dropout & Regularization](#114-dropout--regularization)
12. [Putting It All Together](#12-putting-it-all-together)
13. [Modern Variants & Extensions](#13-modern-variants--extensions)
14. [Resources](#14-resources)

---

## 1. Big Picture

The Transformer is a sequence-to-sequence architecture that replaces recurrence (RNNs/LSTMs) entirely with **attention**. Every token can attend to every other token in a single layer, enabling:

- Parallelism during training (no sequential dependency)
- Long-range dependency capture without vanishing gradients
- Scalability to billions of parameters

High-level data flow:

```
Input tokens
    → Token Embeddings + Positional Encoding
    → Encoder Stack  (N × [Self-Attention → FFN])
    → Decoder Stack  (N × [Masked Self-Attention → Cross-Attention → FFN])
    → Linear + Softmax
    → Output tokens
```

> Notes & code: `01_big_picture/`

---

## 2. Input Pipeline

### 2.1 Tokenization

Converting raw text into a sequence of integer token IDs.

| Method | Description |
|--------|-------------|
| Word-level | One ID per word. Large vocab, many unknowns. |
| Character-level | One ID per character. Tiny vocab, long sequences. |
| BPE (Byte-Pair Encoding) | Merges frequent byte/char pairs iteratively. Used by GPT. |
| WordPiece | Similar to BPE, maximizes language model likelihood. Used by BERT. |
| SentencePiece | Language-agnostic, works on raw Unicode. Used by T5, LLaMA. |

Key concepts: vocabulary size `V`, `[CLS]`/`[SEP]`/`[PAD]`/`[UNK]` special tokens, context length.

> Notes & code: `02_input_pipeline/tokenization/`

---

### 2.2 Token Embeddings

Each token ID is mapped to a dense vector of dimension `d_model` via a learned embedding matrix `E ∈ R^{V × d_model}`.

```
token_id  →  E[token_id]  →  vector of shape (d_model,)
```

The same matrix is often **weight-tied** with the final output projection (reduces parameters, improves generalization).

> Notes & code: `02_input_pipeline/embeddings/`

---

### 2.3 Positional Encoding

Attention is permutation-invariant — it has no built-in notion of order. Positional encoding injects position information.

**Sinusoidal (original paper):**
```
PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
```

**Learned absolute PE:** A trainable matrix of shape `(max_seq_len, d_model)`. Used by BERT, GPT.

**Relative PE (RoPE, ALiBi):** Encode relative distances rather than absolute positions. Better length generalization. Used by LLaMA (RoPE), BLOOM (ALiBi).

> Notes & code: `02_input_pipeline/positional_encoding/`

---

## 3. Attention Mechanism

The core building block. Lets every position gather information from all other positions.

### 3.1 Scaled Dot-Product Attention

Given inputs packed into matrices:
- `Q` (queries) — shape `(seq_len, d_k)`
- `K` (keys)    — shape `(seq_len, d_k)`
- `V` (values)  — shape `(seq_len, d_v)`

```
Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) · V
```

- `Q K^T` computes similarity scores between every query-key pair → shape `(seq_len, seq_len)`
- Division by `sqrt(d_k)` prevents saturation of the softmax for large `d_k`
- Softmax normalizes scores to a probability distribution
- Multiply by `V` to get a weighted sum of values

> Notes & code: `03_attention/scaled_dot_product/`

---

### 3.2 Multi-Head Attention

Run `h` attention heads in parallel, each with its own learned projections:

```
head_i = Attention(Q W_i^Q,  K W_i^K,  V W_i^V)

MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
```

- `W_i^Q, W_i^K ∈ R^{d_model × d_k}`,  `d_k = d_model / h`
- `W_i^V ∈ R^{d_model × d_v}`,  `d_v = d_model / h`
- `W^O ∈ R^{h·d_v × d_model}`

Each head learns to attend to different aspects (syntax, coreference, semantics, …).

> Notes & code: `03_attention/multi_head/`

---

### 3.3 Self-Attention vs Cross-Attention

| Type | Q source | K, V source | Used in |
|------|----------|-------------|---------|
| Self-Attention | Same sequence | Same sequence | Encoder, Decoder (1st sub-layer) |
| Cross-Attention | Decoder states | Encoder output | Decoder (2nd sub-layer) |

> Notes & code: `03_attention/cross_attention/`

---

### 3.4 Causal (Masked) Attention

For autoregressive decoding, position `i` must not attend to positions `> i`. Achieved by adding a mask before softmax:

```
mask[i, j] = -inf   if j > i
             0       otherwise
```

> Notes & code: `03_attention/causal_mask/`

---

## 4. Feed-Forward Network

Applied independently to each position after attention:

```
FFN(x) = max(0, x W_1 + b_1) W_2 + b_2
```

- `W_1 ∈ R^{d_model × d_ff}`,  `W_2 ∈ R^{d_ff × d_model}`
- Typical ratio: `d_ff = 4 × d_model`  (e.g. 512 → 2048)
- Non-linearity: ReLU in the original paper; **GELU** is standard in modern models; **SwiGLU** used in LLaMA

This is where most of the model's parameters live and where "knowledge" is thought to be stored.

> Notes & code: `04_feed_forward/`

---

## 5. Layer Normalization

Stabilizes training by normalizing activations across the feature dimension:

```
LayerNorm(x) = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta
```

- `gamma`, `beta` are learned per-feature scale and shift
- Applied over `d_model` dimension (not batch dimension like BatchNorm)

**Pre-LN vs Post-LN:**
- Original paper uses **Post-LN** (after residual add)
- Modern models use **Pre-LN** (before sub-layer) — more stable training, no warmup needed
- **RMSNorm** (no mean subtraction) used by LLaMA — simpler and faster

> Notes & code: `05_layer_norm/`

---

## 6. Residual Connections

Each sub-layer (attention, FFN) is wrapped with a residual (skip) connection:

```
output = LayerNorm(x + SubLayer(x))   # Post-LN
output = x + SubLayer(LayerNorm(x))   # Pre-LN
```

Benefits:
- Enables gradient flow through deep networks
- Allows the network to learn incremental refinements
- Makes it easy to initialize deeper models stably

> Notes & code: `06_residual_connections/`

---

## 7. The Encoder

One encoder layer = Self-Attention block + FFN block (each with residual + norm).

```
x = LayerNorm(x + MultiHeadSelfAttention(x))
x = LayerNorm(x + FFN(x))
```

The full encoder is `N` such layers stacked (N = 6 in the base model). Output is a contextual representation of the input sequence — every position is enriched with context from all other positions.

> Notes & code: `07_encoder/`

---

## 8. The Decoder

One decoder layer has **three** sub-layers:

1. **Masked Self-Attention** — attends to previously generated tokens (causal mask)
2. **Cross-Attention** — attends to encoder output (`K`, `V` from encoder; `Q` from decoder)
3. **FFN**

```
x = LayerNorm(x + MaskedMultiHeadSelfAttention(x))
x = LayerNorm(x + MultiHeadCrossAttention(x, encoder_output))
x = LayerNorm(x + FFN(x))
```

> Notes & code: `08_decoder/`

---

## 9. Encoder-Decoder Stack

The complete seq2seq architecture from the original paper:

```
Encoder: N=6 layers,  d_model=512,  h=8 heads,  d_ff=2048
Decoder: N=6 layers,  d_model=512,  h=8 heads,  d_ff=2048
```

**Decoder-only models** (GPT, LLaMA): drop the encoder + cross-attention entirely. The decoder attends only to itself with a causal mask. Better for generation tasks.

**Encoder-only models** (BERT): drop the decoder. No causal mask — bidirectional context. Better for classification and understanding tasks.

> Notes & code: `09_full_model/`

---

## 10. Output Head

### 10.1 Linear Projection

Map decoder output `(seq_len, d_model)` to logits over vocabulary `(seq_len, V)`:

```
logits = x @ W_E^T      # weight-tied with embedding matrix
```

### 10.2 Softmax & Sampling Strategies

| Strategy | Description |
|----------|-------------|
| Greedy | Pick `argmax` at each step. Fast, deterministic, suboptimal. |
| Beam Search | Keep top-`k` hypotheses at each step. Better quality for translation. |
| Temperature sampling | Divide logits by `T` before softmax. `T<1` sharper, `T>1` more random. |
| Top-k sampling | Sample from only the top `k` tokens by probability. |
| Top-p (nucleus) | Sample from smallest set of tokens whose cumulative prob ≥ `p`. |

> Notes & code: `10_output_head/`

---

## 11. Training

### 11.1 Loss Function (Cross-Entropy)

```
L = - sum_t log P(y_t | y_{<t}, x)
```

Teacher forcing: ground-truth tokens are fed as decoder input during training (not model predictions).

### 11.2 Label Smoothing

Instead of one-hot targets, use `(1 - eps)` for the correct class and `eps / (V-1)` for others. Prevents overconfident predictions. `eps = 0.1` in the original paper.

### 11.3 Optimizer & Learning Rate Schedule

Adam optimizer with the **warmup + inverse square root decay** schedule from the paper:

```
lr = d_model^{-0.5} * min(step^{-0.5}, step * warmup_steps^{-1.5})
```

Warmup prevents large gradient updates early in training when embeddings are uninitialized.

### 11.4 Dropout & Regularization

- Dropout on attention weights, FFN activations, embeddings
- Rate = 0.1 (base) / 0.3 (large)
- Applied during training only

> Notes & code: `11_training/`

---

## 12. Putting It All Together

End-to-end walkthrough: tokenize a sentence pair → forward pass through encoder-decoder → compute loss → backprop → decode output.

> Notes & code: `12_end_to_end/`

---

## 13. Modern Variants & Extensions

| Model | Key Changes |
|-------|-------------|
| BERT | Encoder-only, bidirectional, masked LM pre-training |
| GPT | Decoder-only, autoregressive, scaled to billions |
| T5 | Encoder-decoder, every task as text-to-text |
| LLaMA | Decoder-only, RoPE, SwiGLU, RMSNorm, GQA |
| Flash Attention | IO-aware exact attention — much faster, same output |
| Sparse / MoE | Replace FFN with Mixture-of-Experts (e.g. Mixtral) |
| RWKV / Mamba | Sub-quadratic alternatives to full attention |

> Notes: `13_variants/`

---

## 14. Resources

- **Paper:** Vaswani et al. (2017) — [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- **Illustrated Transformer:** Jay Alammar's blog post
- **The Annotated Transformer:** Harvard NLP
- **nanoGPT:** Andrej Karpathy's minimal GPT implementation

---

## Repo Structure (evolving)

```
LLM-Architecture/
├── README.md
├── 01_big_picture/
├── 02_input_pipeline/
│   ├── tokenization/
│   ├── embeddings/
│   └── positional_encoding/
├── 03_attention/
│   ├── scaled_dot_product/
│   ├── multi_head/
│   ├── cross_attention/
│   └── causal_mask/
├── 04_feed_forward/
├── 05_layer_norm/
├── 06_residual_connections/
├── 07_encoder/
├── 08_decoder/
├── 09_full_model/
├── 10_output_head/
├── 11_training/
├── 12_end_to_end/
└── 13_variants/
```
