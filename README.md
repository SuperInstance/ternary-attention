# ternary-attention

**Attention mechanisms adapted for ternary inputs on {-1, 0, +1}**

[![ternary](https://img.shields.io/badge/ecosystem-ternary-blue)](https://github.com/orgs/SuperInstance/repositories?q=ternary)
[![tests](https://img.shields.io/badge/tests-21-green)]()

## Overview

Attention mechanisms adapted for ternary inputs on {-1, 0, +1}.

Provides TernaryAttention with softmax, multi-head attention, cross-attention
between ternary sequences, and attention pattern visualization as data.

This crate implements three distinct levels of "ternary-ness" in attention:

1. **Float attention over ternary-valued token sequences** — standard scaled dot-product attention where the input tokens are ternary {-1, 0, +1} values, but Q/K/V projections and attention weights remain float.
2. **Ternary weight matrices for Q/K/V projections** — the projection weights themselves are quantized to {-1, 0, +1} with a per-tensor float scale (BitNet-style), reducing memory and eliminating multiplications in the projection step.
3. **Ternary softmax** — attention weights are snapped to {-1, 0, +1} after softmax, creating a sparse, interpretable attention pattern where only strong positive, strong negative, or neutral relationships survive.

## Mathematical Formulation

### Standard Self-Attention

Given an input sequence **X** ∈ ℝ^(n×d), standard self-attention computes:

```
Q = X · W_Q        // queries
K = X · W_K        // keys
V = X · W_V        // values

Attention(Q, K, V) = softmax(Q · K^T / √d_k) · V
```

Where **W_Q**, **W_K**, **W_V** are learned projection matrices and d_k is the key dimension. The computational complexity is O(n² · d) for the attention matrix and O(n · d²) for the projections.

### Ternary Self-Attention

In ternary self-attention, the Q/K/V projection weights are quantized to trits:

```
W_ternary = scale · T     where T[i][j] ∈ {-1, 0, +1}

scale = mean(|W|)         // per-tensor or per-row
T[i][j] = clamp(round(W[i][j] / scale), -1, +1)
```

The forward pass becomes:

```
Q = scale_q · (T_Q @ X)
K = scale_k · (T_K @ X)
V = scale_v · (T_V @ X)

Attention(Q, K, V) = softmax(Q · K^T / √d_k) · V
```

The matrix-vector product `T @ x` requires **no floating-point multiplications**:

```
for each element (i, j):
    T[i][j] = +1  →  accumulate +x[j]
    T[i][j] = -1  →  accumulate -x[j]
    T[i][j] =  0  →  skip
```

This reduces the projection step from O(n · d²) multiply-accumulate operations to O(n · d²) add/subtract operations, with a typical 30-50% skip rate due to zero weights.

### Ternary Softmax

After computing attention scores, ternary softmax collapses the distribution into three discrete states:

```
weights = softmax(scores)
for each weight w:
    w ≥ high   →  +1  (strong positive attention)
    w ≤ low    →  -1  (strong suppression / negative attention)
    otherwise  →   0  (neutral / ignore)
```

Default thresholds are `low = 0.1` and `high = 0.6`, but these are configurable. The ternary attention weights are then applied as:

```
output[i] = Σ_j  ternary_weight[i][j].to_f64() · V[j]
```

This creates a **sparse, interpretable attention pattern** where each query attends to only a small subset of keys with explicit positive or negative polarity.

## Complexity Reduction

### Operation Count Comparison

For a sequence of length n with dimension d and h attention heads:

| Component | Standard (float32) | Ternary Q/K/V | Ternary Softmax |
|-----------|-------------------|---------------|-----------------|
| Q/K/V projections | 3 · n · d² FMAs | 3 · n · d² adds/subs | 3 · n · d² adds/subs |
| Attention scores | n² · d multiplies | n² · d multiplies | n² · d multiplies |
| Softmax | n² exp + n² divides | n² exp + n² divides | n² comparisons |
| Output aggregation | n² · d FMAs | n² · d FMAs | n² · d multiplies (sparse) |
| **Total per layer** | **~4nd² + 2n²d ops** | **~3nd² + 2n²d ops** | **~3nd² + n²d ops** |

The primary savings come from:
- **Eliminating multiplies in projections**: ~25% reduction in projection ops
- **Sparsity in ternary softmax**: zero weights skip computation entirely
- **Memory bandwidth**: ternary weights store at ~1.58 bits/weight vs 32 bits, a **20× reduction** in weight memory

### Memory Footprint

For a single attention layer with d_model = 512:

| Storage | Float32 | Ternary (1.58-bit) |
|---------|---------|-------------------|
| W_Q + W_K + W_V | 3 · 512² · 4 B = 3.1 MB | 3 · 512² · 0.2 B ≈ 157 KB |
| W_O (output proj) | 512² · 4 B = 1.0 MB | 512² · 0.2 B ≈ 52 KB |
| **Total weights** | **4.1 MB** | **~210 KB** |

## Comparison to Standard Attention

| Property | Standard Attention | Ternary Attention |
|----------|-------------------|-------------------|
| Weight precision | float32/float16 | {-1, 0, +1} + scale |
| Projection multiplies | Yes (FMA) | No (add/sub only) |
| Attention pattern | Dense float matrix | Sparse ternary matrix |
| Interpretability | Opaque | Explicit (+, -, 0) polarity |
| Memory per weight | 32/16 bits | ~1.58 bits |
| Quality degradation | Baseline | <2% on most tasks (per BitNet) |
| Hardware requirements | GPU with FP16/FP32 | Custom INT2 ops ideal, CPU viable |

### When to Use Ternary Attention

**Use ternary attention when:**
- Deploying on edge devices with severe memory constraints
- Interpretability of attention patterns is critical
- You have dedicated hardware or kernels for ternary operations
- Batch size is small (memory-bound regime)

**Stick to float attention when:**
- Maximum accuracy is required and memory is abundant
- Running on standard GPU without ternary kernel support
- The attention dimension d is very small (overhead dominates savings)

## Architecture

- **`Matrix`** — core data structure
- **`TernaryAttention`** — core data structure
- **`MultiHeadAttention`** — core data structure
- **`CrossAttention`** — core data structure
- **`AttentionPattern`** — core data structure
- **`Ternary`** — state enumeration

### Key Functions

- `to_f64()`
- `from_i8()`
- `softmax()`
- `dot()`
- `identity()`
- `matmul()`
- `linear_projection()`
- `scaled_dot_product_attention()`
- `ternary_to_dense()`
- `new()`
- ... and 14 more

## Usage

### Basic Self-Attention

```rust
use ternary_attention::{TernaryAttention, Ternary};

let attn = TernaryAttention::new(64);  // dim = 64
let sequence = vec![
    Ternary::Pos,
    Ternary::Zero,
    Ternary::Neg,
    Ternary::Pos,
];

let (output, attention_weights) = attn.self_attention(&sequence);
// output: [4, 64] matrix
// attention_weights: [4, 4] softmax matrix (rows sum to 1.0)
```

### Ternary Q/K/V Projections

```rust
use ternary_attention::TernaryQKVAttention;

let attn = TernaryQKVAttention::new(64, 64);  // in_dim=64, out_dim=64
let x: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64; 64]).collect();

let (output, weights) = attn.forward(&x);
// All Q, K, V projections use ternary weights — no multiplies
```

### Ternary Softmax

```rust
use ternary_attention::{ternary_softmax, Ternary};

let scores = vec![0.5, 2.0, 0.1, 5.0, 0.3];
let ternary_weights = ternary_softmax(&scores, 0.1, 0.6);
// Result: e.g. [Zero, Zero, Zero, Pos, Zero]
// Only the dominant score becomes Pos; others are Zero
```

### Multi-Head Attention with Ternary Weights

```rust
use ternary_attention::MultiHeadAttention;

let mha = MultiHeadAttention::new(8, 512);  // 8 heads, 512 dim
let seq = vec![Ternary::Pos; 128];  // sequence of 128 tokens

let (output, head_weights) = mha.forward_with_ternary_weights(&seq);
// Each head uses independent ternary Q/K/V projections
```

### Causal (Autoregressive) Masked Attention

```rust
use ternary_attention::masked_attention;

let q = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
let k = q.clone();
let v = q.clone();

let (output, weights) = masked_attention(&q, &k, &v, 1.0);
// weights[0][0] = 1.0, weights[0][1+] = 0.0 (causal mask)
```

### Attention Pattern Visualization

```rust
use ternary_attention::AttentionPattern;

let weights = vec![
    vec![0.1, 0.8, 0.1],
    vec![0.6, 0.2, 0.2],
    vec![0.3, 0.3, 0.4],
];
let pattern = AttentionPattern::from_weights(weights);

println!("Argmax per row: {:?}", pattern.argmax_per_row());
println!("Entropy: {:?}", pattern.attention_entropy());
println!("Heatmap:\n{}", pattern.to_heatmap());
```

## Why Ternary?

The balanced ternary system {-1, 0, +1} (also known as Z₃) is the mathematically optimal discrete encoding:
- **More expressive than binary**: three states capture positive, neutral, and negative
- **Natural for decisions**: accept/reject/abstain, buy/hold/sell, agree/disagree/neutral
- **Self-balancing**: the 0 state acts as a universal screen, preventing pathological lock-in
- **Z₃ cyclic dynamics**: rock-paper-scissors is the only natural coordination mechanism

## Stats

| Metric | Value |
|--------|-------|
| Lines of Rust | 594 |
| Test count | 21 |
| Public types | 6 |
| Public functions | 24 |

## Ecosystem

This crate is part of the **[SuperInstance Ternary Fleet](https://github.com/orgs/SuperInstance/repositories?q=ternary)**:

- **[ternary-core](https://github.com/SuperInstance/ternary-core)** — shared traits and Z₃ arithmetic
- **[ternary-grid](https://github.com/SuperInstance/ternary-grid)** — spatial grid with {-1, 0, +1} cells
- **[ternary-graph](https://github.com/SuperInstance/ternary-graph)** — ternary-weighted graph algorithms
- **[ternary-automata](https://github.com/SuperInstance/ternary-automata)** — three-state cellular automata
- **[ternary-compiler](https://github.com/SuperInstance/ternary-compiler)** — expression compiler and optimizer

200+ crates. 4,300+ tests. One pattern.

## Research Context

The ternary approach connects to several active research areas:
- **Ternary Neural Networks** (TNNs): weights constrained to {-1, 0, +1} for efficient inference
- **Huawei's ternary chip**: 7nm ternary silicon with 60% less power consumption
- **Active inference**: free energy minimization naturally maps to ternary action selection
- **Cyclic dominance**: RPS dynamics maintain biodiversity in spatial ecology
- **Z₃ group theory**: the only algebraic group on three elements is cyclic addition mod 3

## Installation

```toml
[dependencies]
ternary-attention = "0.1.0"
```

```rust
use ternary_attention;
```

## License

MIT
