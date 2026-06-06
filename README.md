# ternary-attention

**Attention mechanisms adapted for ternary inputs on {-1, 0, +1}**

[![ternary](https://img.shields.io/badge/ecosystem-ternary-blue)](https://github.com/orgs/SuperInstance/repositories?q=ternary)
[![tests](https://img.shields.io/badge/tests-21-green)]()

## Overview

Attention mechanisms adapted for ternary inputs on {-1, 0, +1}.

Provides TernaryAttention with softmax, multi-head attention, cross-attention
between ternary sequences, and attention pattern visualization as data.

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

## Usage

```toml
[dependencies]
ternary-attention = "0.1.0"
```

```rust
use ternary_attention;
```

## License

MIT
![Migration](https://img.shields.io/badge/ternary_types-v0.2.0-blueviolet)

## Migration

This crate uses `ternary_types::Ternary` (canonical) instead of its own `enum Ternary`.
