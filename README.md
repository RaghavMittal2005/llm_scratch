# LLM From Scratch

A ground-up implementation of a large language model — from raw transformer math to RLHF — built entirely in PyTorch.

## Setup

```bash
conda create -n llm_from_scratch python=3.11
conda activate llm_from_scratch
pip install -r requirements.txt
```



### Part 0 — Foundations & Mindset
- High-level LLM training pipeline: pretraining → fine-tuning → alignment
- Hardware & software setup (PyTorch, CUDA/MPS, mixed precision, profiling)

### Part 1 — Core Transformer Architecture
- Positional embeddings: absolute learned vs. sinusoidal
- Self-attention from first principles, with a manual worked example
- Single attention head in PyTorch
- Multi-head attention: splitting, concatenation, projections
- Feed-forward networks: GELU, dimensionality expansion
- Residual connections & LayerNorm
- Stacking into a full Transformer block

### Part 2 — Training a Tiny LLM
- Byte-level tokenization
- Dataset batching & next-token prediction targets
- Cross-entropy loss & label shifting
- Training loop from scratch (no Trainer API)
- Sampling strategies: temperature, top-k, top-p
- Validation loss evaluation

### Part 3 — Modernizing the Architecture
- RMSNorm: replacing LayerNorm, comparing gradients & convergence
- RoPE (Rotary Positional Embeddings): theory & implementation
- SwiGLU activations in the MLP
- KV cache for faster inference
- Sliding-window attention & attention sink
- Rolling buffer KV cache for streaming

### Part 4 — Scaling Up
- Switching from byte-level to BPE tokenization
- Gradient accumulation & mixed precision training
- Learning rate schedules with warmup
- Checkpointing & training resumption
- Logging & visualization with TensorBoard / W&B

### Part 5 — Mixture of Experts (MoE)
- Expert routing, gating networks, and load balancing
- Implementing MoE layers in PyTorch
- Hybrid dense + MoE architectures

### Part 6 — Supervised Fine-Tuning (SFT)
- Instruction dataset formatting (prompt + response)
- Causal LM loss with masked labels
- Curriculum learning for instruction data
- Evaluating outputs against gold responses

### Part 7 — Reward Modeling
- Preference datasets and pairwise rankings
- Reward model architecture (transformer encoder)
- Loss functions: Bradley–Terry, margin ranking loss
- Sanity checks for reward shaping

### Part 8 — RLHF with PPO
- **Policy network:** SFT base model with a value head
- **Reward signal:** reward model from Part 7
- **PPO objective:** maximize reward while penalizing KL divergence from SFT policy
- **Training loop:** sample prompts → generate completions → score → optimize
- **Stability tricks:** reward normalization, KL-controlled rollout length, gradient clipping

---

## Philosophy

Every component is built from scratch — no black-box wrappers. The goal is to understand not just *what* each piece does, but *why* it exists and what breaks when you remove it.
