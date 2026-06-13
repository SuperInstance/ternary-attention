# ternary-attention

**Attention mechanisms for balanced ternary sequences in neural architectures.**

`ternary-attention` implements scaled dot-product attention, multi-head attention, and cross-attention specialized for ternary-valued inputs from $\{-1, 0, +1\}$. It provides the mathematical foundations for ternary transformers, enabling structured attention analysis over ternary-encoded information.

## Why It Matters

The attention mechanism is the core innovation behind transformer architectures. By adapting attention to balanced ternary inputs — where each token carries a value in $\{-1, 0, +1\}$ — we enable neural processing of ternary-native data without the lossy projection to continuous embeddings that binary or real-valued attention would require.

Ternary attention has applications in:

- **Ternary neural networks (TNNs):** Networks where weights and activations are quantized to $\{-1, 0, +1\}$, reducing memory by $32\times$ vs. FP32.
- **Balanced ternary computing:** Native computation in base-3 arithmetic systems.
- **Information-theoretic attention:** The three-valued domain naturally models (suppress, ignore, amplify) decisions.

## How It Works

### Scaled Dot-Product Attention

Given query matrix $Q \in \mathbb{R}^{n_q \times d}$, key matrix $K \in \mathbb{R}^{n_k \times d}$, and value matrix $V \in \mathbb{R}^{n_k \times d_v}$, attention computes:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V$$

The scaling factor $\sqrt{d}$ prevents the dot products from growing too large in magnitude, which would push softmax into regions with vanishing gradients.

**Numerically stable softmax:**

$$\text{softmax}(x_i) = \frac{e^{x_i - \max(\mathbf{x})}}{\sum_j e^{x_j - \max(\mathbf{x})}}$$

The subtraction of $\max(\mathbf{x})$ prevents overflow while preserving the probability distribution (the exponential of a shifted distribution is proportional to the original).

**Complexity:** $O(n_q \cdot n_k \cdot d)$ for score computation, $O(n_q \cdot n_k)$ for softmax, $O(n_q \cdot n_k \cdot d_v)$ for the weighted sum. Total: $O(n_q \cdot n_k \cdot d)$.

### Ternary-to-Dense Embedding

Ternary sequences are projected to dense vectors via a deterministic scaling:

$$e_i = v \cdot \frac{i+1}{d}, \qquad v \in \{-1, 0, +1\},\; i \in \{0, \ldots, d-1\}$$

This preserves the sign and magnitude of the ternary value while creating a $d$-dimensional representation suitable for attention computation.

### Multi-Head Attention

Multi-head attention splits the $d$-dimensional representation into $h$ heads of dimension $d_h = d/h$, computes attention independently per head, and concatenates:

$$\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)$$

$$\text{head}_i = \text{Attention}(Q_i, K_i, V_i), \quad Q_i = K_i = V_i = X[:, i \cdot d_h : (i+1) \cdot d_h]$$

Each head attends to a different subspace, enabling the model to capture multiple patterns simultaneously.

**Complexity:** $O(h \cdot n^2 \cdot d_h) = O(n^2 \cdot d)$ — same as single-head attention but with richer representational capacity.

### Cross-Attention

Cross-attention computes attention between two distinct sequences — a source (queries) and a target (keys/values):

$$\text{CrossAttn}(S, T) = \text{softmax}\!\left(\frac{Q_S K_T^\top}{\sqrt{d}}\right) V_T$$

This enables tasks like sequence-to-sequence mapping where the source attends to relevant positions in the target.

### Masked (Causal) Attention

For autoregressive generation, a causal mask prevents attending to future positions:

$$M_{ij} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

After softmax, masked positions receive zero weight, enforcing causality.

### Attention Pattern Analysis

The `AttentionPattern` structure provides:

- **Entropy per row:** $H_i = -\sum_j w_{ij} \log w_{ij}$ — measures how focused vs. diffuse each query's attention is. Low entropy = focused attention; high entropy = diffuse attention.
- **Argmax per row:** The most-attended position for each query.
- **Uniformity test:** Checks whether attention weights are approximately uniform within a tolerance $\epsilon$.

### Ternary Compatibility Function

A lightweight compatibility score between ternary sequences without dense projection:

$$C(\mathbf{q}, \mathbf{k}) = \frac{1}{n} \sum_{i=1}^{n} q_i \cdot k_i$$

where $q_i, k_i \in \{-1, 0, +1\}$. The score ranges from $-1$ (perfectly opposite) to $+1$ (perfectly aligned), with $0$ indicating orthogonality.

## Quick Start

```toml
[dependencies]
ternary-attention = "0.1"
```

```rust
use ternary_attention::{TernaryAttention, MultiHeadAttention, CrossAttention, Ternary};

// Self-attention on a ternary sequence
let attn = TernaryAttention::new(4);
let seq = vec![Ternary::Neg, Ternary::Zero, Ternary::Pos];
let (output, weights) = attn.self_attention(&seq);
assert_eq!(output.len(), 3);
// Each row of attention weights sums to 1.0
for row in &weights {
    let sum: f64 = row.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
}

// Multi-head attention (2 heads, 4-dim)
let mha = MultiHeadAttention::new(2, 4);
let (out, head_weights) = mha.forward(&seq);
assert_eq!(head_weights.len(), 2);

// Cross-attention between two sequences
let ca = CrossAttention::new(4);
let source = vec![Ternary::Neg, Ternary::Pos];
let target = vec![Ternary::Zero, Ternary::Pos, Ternary::Neg];
let (cross_out, cross_w) = ca.forward(&source, &target);
```

## API

| Type/Function | Purpose |
|---------------|---------|
| `Ternary` | The $\{-1, 0, +1\}$ value type |
| `softmax()` | Numerically stable softmax over a slice |
| `scaled_dot_product_attention()` | Core attention function |
| `masked_attention()` | Causal/masked attention variant |
| `TernaryAttention` | Single-head attention over ternary sequences |
| `MultiHeadAttention` | Multi-head attention with $h$ heads |
| `CrossAttention` | Cross-attention between source and target |
| `AttentionPattern` | Weight matrix with analytics (entropy, argmax, heatmap) |
| `ternary_compatibility()` | O(n) compatibility score for raw ternary sequences |
| `matmul()`, `identity()`, `dot()` | Linear algebra primitives |

## Architecture Notes

Attention directly instantiates the SuperInstance conservation law **γ + η = C**. The softmax operation is a **partition function normalization**: it distributes a unit of "attention energy" across positions. Focused attention (low entropy) concentrates $\gamma$ — energy is directed at specific positions, creating structured order. Diffuse attention (high entropy) distributes $\eta$ — energy is spread uniformly, approaching maximum information entropy.

The scaling factor $1/\sqrt{d}$ acts as a **thermodynamic regulator**: without it, large dot products would create near-deterministic attention (all energy concentrated at one position), violating the conservation bound by pushing $\gamma$ beyond $C$.

Multi-head attention partitions the total energy $C$ across $h$ heads, each operating on $C/h$ effective energy. This ensures that no single head can dominate the representational budget.

## References

- Vaswani, A. et al. *Attention Is All You Need.* NeurIPS 2017. — Original transformer attention.
- Bahdanau, D. et al. *Neural Machine Translation by Jointly Learning to Align and Translate.* ICLR 2015. — Additive attention.
- Li, Y. et al. *Ternary Weight Networks.* arXiv:1605.04711, 2016. — Quantization to $\{-1, 0, +1\}$.
- Jaynes, E.T. *Information Theory and Statistical Mechanics.* Physical Review 1957. — Maximum entropy principle.

## License

MIT
