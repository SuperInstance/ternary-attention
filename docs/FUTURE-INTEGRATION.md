# Future Integration: ternary-attention

## Current State
Implements scaled dot-product attention, multi-head attention, and cross-attention over ternary sequences. Provides `TernaryAttention` with linear projections, softmax over attention scores, and attention pattern visualization.

## Integration Opportunities

### With ternary-cell / construct-core
Each attention head becomes a **room perception system**. In room-as-codespace, a `TernaryCell` maintains a `TernaryAttention` module where: queries = current cell state, keys = neighbor cell states, values = neighbor payloads. The attention weights determine which neighboring rooms a cell "listens to" during its `tick()` cycle. `MultiHeadAttention` maps to multiple perception channels (e.g., thermal, acoustic, occupancy) processed in parallel.

### With ternary-transform
The wavelet transform from `TernaryWavelet::forward()` can serve as a preprocessing step before attention — decomposing room state into multi-resolution approximation/detail coefficients. Attention then operates on the wavelet domain rather than raw state, giving cells multi-scale awareness of their neighborhood.

### With ternary-rl
Replace the `QTable`'s greedy `best_action()` with attention-weighted action selection. The attention scores over state history determine which past experiences inform the current action, creating an experience-weighted RL agent.

## Potential in Mature Systems
In PLATO's tiered architecture, attention runs at Layer 1 (PiConstruct): a `SyncConstruct` uses `scaled_dot_product_attention()` to route incoming `TernaryMessenger` signals based on learned relevance. At Layer 0 (ESP32), the softmax collapses to a hard argmax — pure ternary routing with no floating point. The `cross_attention()` function becomes the mechanism by which different construct types share context without merging state.

## Cross-Pollination Ideas
**Music × Attention:** Voice leading in ternary-music is structurally identical to cross-attention. Each voice is a query sequence; the target chord is the key/value sequence. The attention pattern IS the voice-leading map — which note in voice A moves to which note in chord B. The `softmax` weights encode the smoothness of each voice-leading choice. This connects directly to the PLR group from `flux-algebra-rs`.

**Game theory × Attention:** In cooperative games, Shapley values approximate attention weights. Ternary attention over agent coalitions could compute fast Shapley approximations.

## Dependencies for Next Steps
- `ternary-cell` must expose its tick cycle as a trait for attention injection
- `ternary-tensor` needed for batched attention on large cell grids
- Benchmark against floating-point attention to quantify the information loss from ternary discretization
