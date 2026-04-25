# Introduction to Transformers — Part 1

**Lecture:** Introduction to Transformers | Transformers Part 1  
**Video:** https://www.youtube.com/watch?v=BjRVS2wTtcA

---

## 1. Historical Timeline

| Year | Development |
|------|-------------|
| 2000–2014 | RNNs and LSTMs dominate NLP. All major tasks — translation, summarization, sentiment — are built on sequential recurrent models. |
| 2014 | **Encoder-Decoder with LSTM** introduced for seq2seq tasks. The encoder compresses the whole input into a single fixed-size vector; the decoder reads that vector to generate output. Works, but the fixed-size bottleneck hurts long sequences. |
| 2015–2016 | **Attention mechanism** proposed. Instead of one fixed vector, the decoder can look at all encoder hidden states and dynamically decide which parts of the input matter most at each decoding step. Big improvement, but attention is still bolted on top of LSTMs. |
| 2017 | **"Attention Is All You Need"** (Vaswani et al.). Removes the LSTM entirely. The whole architecture is built on self-attention. Enables full parallelism and scales to much larger models. |
| 2018 | **BERT** (Google) and **GPT** (OpenAI) show that pre-training a large Transformer on raw text and then fine-tuning on specific tasks beats every prior approach. Transfer learning era begins. |
| 2018–2020 | Transformers move beyond NLP — **Vision Transformer (ViT)** for images, **AlphaFold 2** (DeepMind) uses Transformers to predict protein 3D structure, achieving a 50-year-old biology grand challenge. |
| 2021–2023 | **Generative AI** era — ChatGPT, DALL·E 2, Midjourney, Stable Diffusion, RunwayML. Transformers generate text, images, audio, and video at human-level quality. |

---

## 2. Why Transformers? The Problem with RNNs

To understand why Transformers matter, you need to understand what they replaced.

### RNN / LSTM Limitations

**1. Sequential processing (no parallelism)**
- An RNN processes token 1 → token 2 → token 3, one at a time
- Each step depends on the previous hidden state
- You cannot process step 5 until steps 1–4 are done
- Result: training is slow; you can't fully utilise modern GPUs which excel at parallel operations

**2. Vanishing gradient problem**
- During backpropagation through time, gradients shrink exponentially as they travel back through long sequences
- The model effectively "forgets" what happened early in the sequence
- LSTMs mitigate this with gates but don't fully solve it for very long contexts

**3. Fixed-size context vector bottleneck**
- The encoder reads the entire input and compresses it into a single fixed-size vector
- That one vector must carry all information about the input to the decoder
- For long or complex sentences, a lot of information gets lost in this compression

### How Attention Helped (But Wasn't Enough)
- Attention (2015) let the decoder look back at all encoder hidden states, not just the final one
- Solved the bottleneck problem — no more single fixed vector
- But the LSTM was still there, so sequential processing remained
- Transformers replaced the entire RNN with attention, solving all three problems at once

---

## 3. What is a Transformer?

A Transformer is a **sequence-to-sequence neural network** built entirely on attention — no recurrence, no convolution.

### Core Idea
Instead of processing tokens one by one, a Transformer processes the **entire sequence at once**. Every token directly attends to every other token in a single step.

### High-Level Architecture

```
Input sequence  →  [Token Embeddings + Positional Encoding]
                              ↓
                         ENCODER
              ┌─────────────────────────────┐
              │  Multi-Head Self-Attention   │  ← each token attends to all others
              │  +  Feed-Forward Network    │  ← position-wise transformation
              │  (repeated N times)         │
              └─────────────────────────────┘
                              ↓
                    Context representations
                              ↓
                         DECODER
              ┌──────────────────────────────────┐
              │  Masked Multi-Head Self-Attention │  ← attends only to past tokens
              │  +  Cross-Attention              │  ← attends to encoder output
              │  +  Feed-Forward Network         │
              │  (repeated N times)              │
              └──────────────────────────────────┘
                              ↓
                   Linear + Softmax → Output token
```

### Encoder vs Decoder

| | Encoder | Decoder |
|--|---------|---------|
| Purpose | Understand the input | Generate the output |
| Attention type | Bidirectional self-attention (every token sees all others) | Masked self-attention (each token only sees past tokens) + cross-attention to encoder |
| Used in | BERT-style models | GPT-style models, seq2seq translation |

---

## 4. Key Technical Innovations

### Self-Attention
- The mechanism that makes it all work
- For each token, compute how much it should "attend to" every other token
- Produces a weighted sum of all token representations
- Replaces the sequential hidden state passing of RNNs

### Multi-Head Attention
- Run self-attention multiple times in parallel with different learned projections
- Each "head" can focus on a different aspect — one head might track syntax, another coreference, another semantics
- Results are concatenated and projected back

### Positional Encoding
- Since attention has no built-in notion of order, position info is injected via sinusoidal or learned position embeddings added to token embeddings

### Residual Connections
- Each sub-layer (attention, FFN) adds its output back to its input: `output = x + SubLayer(x)`
- Allows gradients to flow directly through the network without vanishing
- Lets the model learn incremental refinements rather than full transformations

### Layer Normalization
- Normalizes activations across the feature dimension within each layer
- Stabilises training and speeds up convergence

### Feed-Forward Network (FFN)
- A small two-layer MLP applied independently to each token position after attention
- Expands the dimension (e.g. 512 → 2048) then projects back
- Where most of the model's "knowledge" is thought to be stored

### Transfer Learning
- Pre-train on massive unlabelled text (next-token prediction or masked LM)
- Fine-tune on small labelled datasets for specific tasks
- Democratises AI — you don't need to train from scratch for every task

---

## 5. Impact of Transformers

- **Revolutionised NLP** — state-of-the-art on translation, summarisation, Q&A, classification simultaneously
- **Democratised AI** — BERT, GPT models are publicly available; small teams can fine-tune powerful models cheaply
- **Unified architecture** — same Transformer backbone used for text, images, audio, video, proteins, code, games
- **Accelerated Generative AI** — ChatGPT, DALL·E, Stable Diffusion, Copilot all built on Transformer foundations

---

## 6. Advantages

| Advantage | Detail |
|-----------|--------|
| Scalability | Fully parallelisable — scales to billions/trillions of parameters with more GPUs |
| Transfer learning | One pre-trained model → many tasks via fine-tuning |
| Long-range dependencies | Any token can directly attend to any other token regardless of distance |
| Multimodal | Text, images, audio, video — same architecture with different input tokenisation |
| Rich ecosystem | Hugging Face Transformers library, PyTorch, JAX/Flax, thousands of pre-trained checkpoints |
| Composability | Works with GANs, RL, retrieval systems, tool use |

---

## 7. Disadvantages

| Disadvantage | Detail |
|--------------|--------|
| Quadratic attention cost | Attention computes pairwise scores for all token pairs → O(n²) memory and compute with sequence length |
| Data hungry | Needs billions of tokens for effective pre-training |
| Expensive hardware | Multi-GPU / TPU setups required for large model training |
| Energy consumption | Training GPT-3 scale models costs millions of dollars and enormous electricity |
| Black box | Hard to explain why a specific output was generated — risky in healthcare, legal, finance |
| Bias & ethics | Training on internet data inherits societal biases; data consent issues |

---

## 8. Applications

| Domain | Application | Examples |
|--------|-------------|----------|
| NLP | Machine translation, summarisation, Q&A | Google Translate, ChatGPT |
| Conversational AI | Human-like dialogue | ChatGPT, Claude, Gemini |
| Image generation | Text-to-image | DALL·E 2, Midjourney, Stable Diffusion |
| Science | Protein structure prediction | AlphaFold 2 |
| Code | Code generation and completion | GitHub Copilot, Codex |
| Multimodal | Text + image + audio | GPT-4V, Gemini, RunwayML |
| Video | Text-to-video generation | Sora, RunwayML |

---

## 9. Future Directions

- **Efficiency** — Flash Attention, sparse attention, linear attention to reduce the O(n²) cost; pruning, quantisation, and distillation to shrink model size
- **Long context** — extending context windows (4K → 128K → 1M+ tokens) via RoPE, ALiBi, and retrieval augmentation
- **Domain-specific models** — LegalGPT, BioGPT, FinGPT trained on domain corpora
- **Multilingual** — models covering low-resource languages beyond English
- **Interpretability** — mechanistic interpretability research to understand what attention heads and neurons actually do
- **Responsible AI** — bias detection and mitigation, data provenance, consent frameworks
- **Sub-quadratic architectures** — Mamba, RWKV, SSMs as potential Transformer replacements for very long sequences

---

## Key Terms

| Term | Definition |
|------|-----------|
| **Transformer** | Attention-only seq2seq architecture introduced in 2017 |
| **Self-Attention** | Mechanism where each token computes a weighted sum over all other tokens |
| **Multi-Head Attention** | Running attention in parallel with multiple learned projections |
| **Encoder** | Transformer component that builds contextual representations of the input |
| **Decoder** | Transformer component that generates output tokens autoregressively |
| **Positional Encoding** | Adds position information to token embeddings since attention is order-agnostic |
| **Residual Connection** | Skip connection: `output = x + SubLayer(x)` for gradient flow |
| **Layer Normalization** | Normalises activations within a layer to stabilise training |
| **Transfer Learning** | Pre-train on large data, fine-tune on small task-specific data |
| **LLM** | Large Language Model — a large Transformer pre-trained on text |
| **Generative AI** | Models that generate new content (text, image, audio, video) |
| **Seq2Seq** | Sequence-to-sequence — maps an input sequence to an output sequence |
