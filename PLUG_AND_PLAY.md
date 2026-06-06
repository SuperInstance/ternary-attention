# PLUG_AND_PLAY — Attention

> Ternary attention mechanism for transformer models

## 🚀 Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
ternary-attention = { git = "https://github.com/SuperInstance/ternary-attention" }
```

Use in your code:

```rust
use ternary_attention::TernaryAttention;

let mut attn = TernaryAttention::new(64, 8);
let out = attn.forward(&queries, &keys, &values);
```

## 📚 Available Documentation

| Document | Description |
|----------|-------------|
| `docs/FROM_BINARY.md` | Understanding ternary concepts as a binary programmer |
| `docs/MIGRATION.md` | Version migration guide |
| `docs/FUTURE-INTEGRATION.md` | Planned features and roadmap |

## 🔗 Integration

This crate is part of the [SuperInstance ternary fleet](https://github.com/SuperInstance). It uses the canonical `Ternary` type from `ternary-types` for cross-crate compatibility.

## 📄 License

MIT
