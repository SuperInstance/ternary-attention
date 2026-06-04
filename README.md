# ternary-attention

Attention mechanisms for ternary-valued sequences — scaled dot-product attention, multi-head attention, cross-attention, causal masking, and attention pattern analysis on {-1, 0, +1} inputs.

## Why This Exists

The transformer architecture revolutionized machine learning by learning *where to look* through attention mechanisms. But standard attention operates on continuous floating-point weights, which is overkill when your inputs are naturally ternary — sentiment (-1/0/+1), market signals (bearish/neutral/bullish), or ternary logic values (false/unknown/true).

This crate implements attention mechanisms specifically designed for ternary input sequences. The ternary compatibility function scores query-key pairs by their element-wise product: matching signs reinforce, opposing signs cancel, and zeros contribute nothing. This produces naturally interpretable attention patterns where the model attends to sequences with similar ternary structure.

The crate includes full matrix operations (softmax, matmul, linear projection), scaled dot-product attention, multi-head attention with configurable head count, cross-attention between different-length sequences, causal masking for autoregressive use, and an `AttentionPattern` type with heatmap visualization and entropy analysis.

This crate is part of the **Negative Space Intelligence** ecosystem.

## Core Concepts

- **Ternary** — A ternary value: `Neg` (-1), `Zero` (0), or `Pos` (+1).
- **TernaryAttention** — Self-attention on ternary sequences. Converts ternary tokens to dense vectors and computes scaled dot-product attention.
- **MultiHeadAttention** — Splits the embedding dimension across multiple attention heads, each computing independent attention, then concatenates results.
- **CrossAttention** — Queries from a source sequence attend to keys/values from a target sequence (different lengths supported).
- **AttentionPattern** — Structured attention weights with argmax-per-row, entropy computation, uniformity checking, and ASCII heatmap rendering.
- **Masked Attention** — Causal (autoregressive) attention where position i can only attend to positions ≤ i.
- **Ternary Compatibility** — A simple scoring function: Σ(qᵢ × kᵢ) / n, producing +1 for full agreement, -1 for full opposition.

## Quick Start

```toml
# Cargo.toml
[dependencies]
ternary-attention = "0.1"
```

```rust
use ternary_attention::*;

// Self-attention on a ternary sequence
let attn = TernaryAttention::new(4);
let seq = vec![Ternary::Neg, Ternary::Zero, Ternary::Pos];
let (output, weights) = attn.self_attention(&seq);
assert_eq!(output.len(), 3);
// Each weight row sums to 1.0 (softmax normalization)

// Multi-head attention
let mha = MultiHeadAttention::new(2, 4); // 2 heads, dim 4
let seq = vec![Ternary::Neg, Ternary::Zero, Ternary::Pos, Ternary::Pos];
let (output, head_weights) = mha.forward(&seq);
assert_eq!(head_weights.len(), 2); // one weight matrix per head

// Cross-attention between two sequences
let ca = CrossAttention::new(4);
let source = vec![Ternary::Neg, Ternary::Pos];
let target = vec![Ternary::Zero, Ternary::Pos, Ternary::Neg];
let (output, weights) = ca.forward(&source, &target);
assert_eq!(output.len(), 2);       // one output per source token
assert_eq!(weights[0].len(), 3);   // attention over 3 target tokens

// Attention pattern analysis
let pattern = AttentionPattern::from_weights(weights);
let focus = pattern.argmax_per_row();       // most attended position per source
let entropy = pattern.attention_entropy();  // sharpness of attention
println!("{}", pattern.to_heatmap());        // ASCII visualization

// Ternary compatibility scoring
let q = vec![Ternary::Pos, Ternary::Pos];
let k = vec![Ternary::Neg, Ternary::Neg];
assert_eq!(ternary_compatibility(&q, &k), -1.0); // full opposition

// Causal (masked) attention for autoregressive generation
let q = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
let (output, weights) = masked_attention(&q, &q, &q, 1.0);
assert!((weights[0][1]).abs() < 1e-6); // position 0 can't see position 1
```

## API Overview

### Core Functions
| Function | Description |
|---|---|
| `softmax(scores)` | Numerically stable softmax |
| `dot(a, b)` | Vector dot product |
| `matmul(a, b)` | Matrix multiplication |
| `linear_projection(input, weight, bias)` | Linear transform |
| `scaled_dot_product_attention(Q, K, V, scale)` | Returns (output, weights) |
| `masked_attention(Q, K, V, scale)` | Causal-masked attention |
| `ternary_compatibility(query, key)` | Ternary pair similarity (-1 to +1) |
| `ternary_to_dense(seq, dim)` | Convert ternary tokens to dense vectors |

### Attention Modules
| Type | Description |
|---|---|
| `TernaryAttention` | Single-head self-attention on ternary sequences |
| `MultiHeadAttention` | Parallel attention heads with concatenation |
| `CrossAttention` | Source queries attend to target keys/values |
| `AttentionPattern` | Weight matrix with analysis and visualization |

## How It Works

Ternary sequences are converted to dense vectors via positional scaling: each ternary value v ∈ {-1, 0, +1} is expanded into a `dim`-length vector where element i equals `v × (i+1)/dim`. This preserves the sign while creating enough dimensional variation for attention to differentiate positions.

Scaled dot-product attention follows the standard formulation: Attention(Q,K,V) = softmax(QKᵀ/√d) · V. The softmax ensures each query's attention weights form a probability distribution. The scale factor √d prevents gradient saturation in the score space.

Multi-head attention splits the embedding dimension into `n_heads` equal parts, runs independent attention in each subspace, and concatenates the results. This allows the model to attend to different aspects of the ternary structure simultaneously — one head might focus on positive/negative contrast while another tracks zero positions.

The attention pattern type provides analysis tools: `argmax_per_row` shows the most-attended position for each query, `attention_entropy` measures how focused (low entropy) or diffuse (high entropy) the attention is, and `to_heatmap` renders a Unicode block-character visualization for debugging.

## Use Cases

1. **Ternary sequence modeling** — Apply transformer-style attention to discrete ternary signals (sentiment flows, market regimes, sensor readings) without converting to continuous embeddings first.

2. **Cross-modal alignment** — Use cross-attention to find correspondences between two ternary sequences of different lengths (e.g., aligning market signals to economic indicators).

3. **Attention visualization** — The `AttentionPattern` type with heatmaps and entropy metrics provides interpretable attention analysis for research and debugging.

4. **Autoregressive ternary generation** — Causal masking enables left-to-right generation of ternary sequences, useful for signal prediction or ternary language modeling.

## Ecosystem

| Crate | Relationship |
|---|---|
| `ternary-network` | Attention weights form a bipartite ternary graph |
| `ternary-logic` | Attention compatibility uses ternary conjunction logic |
| `ternary-bayesian` | Bayesian updates can weight attention by belief |
| `ternary-cell` | Cell signal propagation is a form of local attention |
| `ternary-quantum` | Qutrit gates use similar matrix operations |

## Known Limitations

- **Attention weights are fixed (non-learnable).** The `ternary_to_dense` projection is deterministic — there are no trainable parameters. This is a *simulation* of attention, not a model that can be trained via backpropagation.
- **No gradient computation or backpropagation.** The crate cannot be used for training transformer models.
- **Multi-head attention sees scaled copies of the same information.** Since all heads derive from the same fixed `ternary_to_dense` mapping, the multi-head structure doesn't provide the representational diversity of learned attention heads.
- **`masked_attention` uses `f64::NEG_INFINITY / 2.0` as a validity threshold.** This hack could produce incorrect results with sufficiently negative actual scores.
- **No batch dimension.** All operations work on single sequences only.

## License

MIT
