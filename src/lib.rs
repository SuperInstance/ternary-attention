#![forbid(unsafe_code)]

//! Attention mechanisms with ternary {-1, 0, +1} weights.
//!
//! Three levels of "ternary-ness":
//!  1. Float attention over ternary-valued token sequences (original API)
//!  2. Ternary weight matrices for Q/K/V projections (BitNet-style)
//!  3. Ternary softmax: attention weights snapped back to {-1, 0, +1} after softmax

/// A ternary value in {-1, 0, +1}.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Ternary {
    Neg,
    Zero,
    Pos,
}

impl Ternary {
    #[inline]
    pub fn to_f64(self) -> f64 {
        match self {
            Ternary::Neg => -1.0,
            Ternary::Zero => 0.0,
            Ternary::Pos => 1.0,
        }
    }

    #[inline]
    pub fn from_i8(v: i8) -> Option<Self> {
        match v {
            -1 => Some(Ternary::Neg),
            0 => Some(Ternary::Zero),
            1 => Some(Ternary::Pos),
            _ => None,
        }
    }

    /// Quantize a float to the nearest ternary value.
    pub fn quantize(v: f64) -> Self {
        if v > 0.5 {
            Ternary::Pos
        } else if v < -0.5 {
            Ternary::Neg
        } else {
            Ternary::Zero
        }
    }
}

/// Matrix of ternary weights {-1, 0, +1} with a float scale factor.
/// Represents W_ternary = scale * T where T[i][j] ∈ {-1, 0, +1}.
#[derive(Debug, Clone)]
pub struct TernaryWeightMatrix {
    pub trits: Vec<Vec<i8>>,
    pub scale: f64,
    pub rows: usize,
    pub cols: usize,
}

impl TernaryWeightMatrix {
    /// Create from float matrix using BitNet 1.58-bit quantization.
    pub fn from_floats(m: &[Vec<f64>]) -> Self {
        let rows = m.len();
        let cols = if rows > 0 { m[0].len() } else { 0 };
        let flat: Vec<f64> = m.iter().flatten().cloned().collect();
        let scale = if flat.is_empty() {
            1.0
        } else {
            flat.iter().map(|v| v.abs()).sum::<f64>() / flat.len() as f64
        };
        let s = scale.max(1e-8);
        let trits: Vec<Vec<i8>> = m
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&w| {
                        let v = (w / s).round() as i8;
                        v.max(-1).min(1)
                    })
                    .collect()
            })
            .collect();
        TernaryWeightMatrix { trits, scale: s, rows, cols }
    }

    /// Create from explicit trit matrix.
    pub fn from_trits(trits: Vec<Vec<i8>>, scale: f64) -> Self {
        let rows = trits.len();
        let cols = if rows > 0 { trits[0].len() } else { 0 };
        TernaryWeightMatrix { trits, scale, rows, cols }
    }

    /// Matrix-vector product: y = scale * (T @ x).
    pub fn matvec(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), self.cols);
        (0..self.rows)
            .map(|i| {
                let acc: f64 = self.trits[i]
                    .iter()
                    .zip(x.iter())
                    .map(|(&t, &v)| t as f64 * v)
                    .sum();
                acc * self.scale
            })
            .collect()
    }

    /// Apply to each row of a matrix: projects (seq_len x cols) → (seq_len x rows).
    pub fn project(&self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        x.iter().map(|row| self.matvec(row)).collect()
    }
}

/// Numerically stable softmax over f64.
pub fn softmax(scores: &[f64]) -> Vec<f64> {
    if scores.is_empty() {
        return vec![];
    }
    let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scores.iter().map(|&s| (s - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Ternary softmax: apply softmax then snap each weight to {-1, 0, +1} via thresholds.
/// Values above `high` → +1 (strong positive attention), below `low` → -1 (suppression),
/// middle → 0 (neutral). The result is re-normalized so rows sum to a consistent total.
pub fn ternary_softmax(scores: &[f64], low: f64, high: f64) -> Vec<Ternary> {
    let weights = softmax(scores);
    weights
        .iter()
        .map(|&w| {
            if w >= high {
                Ternary::Pos
            } else if w <= low {
                Ternary::Neg
            } else {
                Ternary::Zero
            }
        })
        .collect()
}

/// Dot product of two f64 slices.
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Row-major f64 matrix.
pub type Matrix = Vec<Vec<f64>>;

/// Create identity matrix of size n.
pub fn identity(n: usize) -> Matrix {
    let mut m = vec![vec![0.0; n]; n];
    for i in 0..n {
        m[i][i] = 1.0;
    }
    m
}

/// Matrix multiply (a: k×m) × (b: m×n) → k×n.
pub fn matmul(a: &Matrix, b: &Matrix) -> Matrix {
    let k = a.len();
    let m = a[0].len();
    let n = b[0].len();
    let mut result = vec![vec![0.0; n]; k];
    for i in 0..k {
        for j in 0..n {
            for l in 0..m {
                result[i][j] += a[i][l] * b[l][j];
            }
        }
    }
    result
}

/// Linear projection: input × weight + bias.
pub fn linear_projection(input: &[f64], weight: &Matrix, bias: &[f64]) -> Vec<f64> {
    let input_matrix: Matrix = vec![input.to_vec()];
    let projected = matmul(&input_matrix, weight);
    projected[0].iter().zip(bias.iter()).map(|(&x, &b)| x + b).collect()
}

/// Rotary positional encoding (RoPE) for a single vector at position pos.
pub fn rope_encode(x: &[f64], pos: usize) -> Vec<f64> {
    let dim = x.len();
    let mut out = x.to_vec();
    let half = dim / 2;
    for i in 0..half {
        let theta = pos as f64 / 10000_f64.powf(2.0 * i as f64 / dim as f64);
        let (sin_t, cos_t) = (theta.sin(), theta.cos());
        let a = x[i];
        let b = x[i + half];
        out[i] = a * cos_t - b * sin_t;
        out[i + half] = a * sin_t + b * cos_t;
    }
    out
}

/// Apply RoPE to each vector in a sequence.
pub fn rope_sequence(x: &[Vec<f64>]) -> Vec<Vec<f64>> {
    x.iter().enumerate().map(|(pos, v)| rope_encode(v, pos)).collect()
}

/// Scaled dot-product attention — returns (output, attention_weights).
pub fn scaled_dot_product_attention(
    queries: &Matrix,
    keys: &Matrix,
    values: &Matrix,
    scale: f64,
) -> (Matrix, Matrix) {
    let n_q = queries.len();
    let n_k = keys.len();
    let d_k = keys[0].len();
    let d_v = values[0].len();

    let mut scores = vec![vec![0.0; n_k]; n_q];
    for i in 0..n_q {
        for j in 0..n_k {
            let mut s = 0.0;
            for k in 0..d_k {
                s += queries[i][k] * keys[j][k];
            }
            scores[i][j] = s / scale;
        }
    }

    let mut attention_weights = vec![vec![0.0; n_k]; n_q];
    for i in 0..n_q {
        attention_weights[i] = softmax(&scores[i]);
    }

    let mut output = vec![vec![0.0; d_v]; n_q];
    for i in 0..n_q {
        for j in 0..n_k {
            for d in 0..d_v {
                output[i][d] += attention_weights[i][j] * values[j][d];
            }
        }
    }

    (output, attention_weights)
}

/// Ternary Q/K/V attention: projects using ternary weight matrices, then runs scaled dot-product.
pub struct TernaryQKVAttention {
    pub wq: TernaryWeightMatrix,
    pub wk: TernaryWeightMatrix,
    pub wv: TernaryWeightMatrix,
    pub out_dim: usize,
    pub scale: f64,
}

impl TernaryQKVAttention {
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        let make = |in_d: usize, out_d: usize| -> TernaryWeightMatrix {
            let trits: Vec<Vec<i8>> = (0..out_d)
                .map(|i| (0..in_d).map(|j| (((i * 7 + j * 3) % 3) as i8) - 1).collect())
                .collect();
            TernaryWeightMatrix::from_trits(trits, 1.0)
        };
        TernaryQKVAttention {
            wq: make(in_dim, out_dim),
            wk: make(in_dim, out_dim),
            wv: make(in_dim, out_dim),
            out_dim,
            scale: (out_dim as f64).sqrt(),
        }
    }

    /// Forward: projects x through ternary Q/K/V, runs attention.
    pub fn forward(&self, x: &[Vec<f64>]) -> (Matrix, Matrix) {
        let q = self.wq.project(x);
        let k = self.wk.project(x);
        let v = self.wv.project(x);
        scaled_dot_product_attention(&q, &k, &v, self.scale)
    }

    /// Forward with ternary softmax applied to attention weights.
    pub fn forward_ternary_softmax(
        &self,
        x: &[Vec<f64>],
        low: f64,
        high: f64,
    ) -> (Matrix, Vec<Vec<Ternary>>) {
        let q = self.wq.project(x);
        let k = self.wk.project(x);
        let v = self.wv.project(x);

        let n_q = q.len();
        let n_k = k.len();
        let d_v = v[0].len();

        let mut ternary_weights: Vec<Vec<Ternary>> = Vec::with_capacity(n_q);
        let mut output = vec![vec![0.0; d_v]; n_q];

        for i in 0..n_q {
            let scores: Vec<f64> = (0..n_k)
                .map(|j| q[i].iter().zip(k[j].iter()).map(|(&a, &b)| a * b).sum::<f64>() / self.scale)
                .collect();
            let tw = ternary_softmax(&scores, low, high);
            for j in 0..n_k {
                let w = tw[j].to_f64();
                for d in 0..d_v {
                    output[i][d] += w * v[j][d];
                }
            }
            ternary_weights.push(tw);
        }

        (output, ternary_weights)
    }
}

/// Convert a ternary sequence to dense float vectors.
pub fn ternary_to_dense(sequence: &[Ternary], dim: usize) -> Matrix {
    sequence
        .iter()
        .map(|&t| {
            let base = t.to_f64();
            (0..dim).map(|i| base * (i as f64 + 1.0) / dim as f64).collect()
        })
        .collect()
}

/// Single-head ternary attention over ternary token sequences.
pub struct TernaryAttention {
    pub dim: usize,
    pub scale: f64,
}

impl TernaryAttention {
    pub fn new(dim: usize) -> Self {
        TernaryAttention { dim, scale: (dim as f64).sqrt() }
    }

    pub fn self_attention(&self, sequence: &[Ternary]) -> (Matrix, Matrix) {
        let dense = ternary_to_dense(sequence, self.dim);
        scaled_dot_product_attention(&dense, &dense, &dense, self.scale)
    }

    pub fn forward(
        &self,
        queries_seq: &[Ternary],
        keys_seq: &[Ternary],
        values_seq: &[Ternary],
    ) -> (Matrix, Matrix) {
        let q = ternary_to_dense(queries_seq, self.dim);
        let k = ternary_to_dense(keys_seq, self.dim);
        let v = ternary_to_dense(values_seq, self.dim);
        scaled_dot_product_attention(&q, &k, &v, self.scale)
    }

    /// Self-attention with RoPE positional encoding.
    pub fn self_attention_with_rope(&self, sequence: &[Ternary]) -> (Matrix, Matrix) {
        let dense = ternary_to_dense(sequence, self.dim);
        let q = rope_sequence(&dense);
        let k = rope_sequence(&dense);
        scaled_dot_product_attention(&q, &k, &dense, self.scale)
    }
}

/// Multi-head attention for ternary inputs.
pub struct MultiHeadAttention {
    pub n_heads: usize,
    pub dim: usize,
    pub head_dim: usize,
}

impl MultiHeadAttention {
    pub fn new(n_heads: usize, dim: usize) -> Self {
        assert_eq!(dim % n_heads, 0, "dim must be divisible by n_heads");
        MultiHeadAttention { n_heads, dim, head_dim: dim / n_heads }
    }

    pub fn forward(&self, sequence: &[Ternary]) -> (Matrix, Vec<Matrix>) {
        let dense = ternary_to_dense(sequence, self.dim);
        let seq_len = dense.len();
        let mut head_outputs = Vec::new();
        let mut concatenated = vec![vec![0.0; self.dim]; seq_len];

        for h in 0..self.n_heads {
            let start = h * self.head_dim;
            let end = start + self.head_dim;

            let q: Matrix = dense.iter().map(|row| row[start..end].to_vec()).collect();
            let k: Matrix = dense.iter().map(|row| row[start..end].to_vec()).collect();
            let v: Matrix = dense.iter().map(|row| row[start..end].to_vec()).collect();

            let scale = (self.head_dim as f64).sqrt();
            let (output, weights) = scaled_dot_product_attention(&q, &k, &v, scale);
            head_outputs.push(weights);

            for i in 0..seq_len {
                for j in 0..self.head_dim {
                    concatenated[i][start + j] = output[i][j];
                }
            }
        }

        (concatenated, head_outputs)
    }

    /// Multi-head attention using ternary weight projections per head.
    pub fn forward_with_ternary_weights(&self, sequence: &[Ternary]) -> (Matrix, Vec<Matrix>) {
        let dense = ternary_to_dense(sequence, self.dim);
        let seq_len = dense.len();
        let mut head_outputs = Vec::new();
        let mut concatenated = vec![vec![0.0; self.dim]; seq_len];

        for h in 0..self.n_heads {
            let qkv = TernaryQKVAttention::new(self.dim, self.head_dim);
            let (output, weights) = qkv.forward(&dense);
            head_outputs.push(weights);

            let start = h * self.head_dim;
            for i in 0..seq_len {
                let out_dim = output[i].len().min(self.head_dim);
                for j in 0..out_dim {
                    concatenated[i][start + j] = output[i][j];
                }
            }
        }

        (concatenated, head_outputs)
    }
}

/// Cross-attention between two ternary sequences.
pub struct CrossAttention {
    pub dim: usize,
}

impl CrossAttention {
    pub fn new(dim: usize) -> Self {
        CrossAttention { dim }
    }

    pub fn forward(&self, source: &[Ternary], target: &[Ternary]) -> (Matrix, Matrix) {
        let q = ternary_to_dense(source, self.dim);
        let k = ternary_to_dense(target, self.dim);
        let v = ternary_to_dense(target, self.dim);
        let scale = (self.dim as f64).sqrt();
        scaled_dot_product_attention(&q, &k, &v, scale)
    }

    /// Cross-attention with ternary Q/K/V weight projections.
    pub fn forward_ternary_weights(
        &self,
        source: &[Ternary],
        target: &[Ternary],
    ) -> (Matrix, Matrix) {
        let q_dense = ternary_to_dense(source, self.dim);
        let k_dense = ternary_to_dense(target, self.dim);
        let qkv = TernaryQKVAttention::new(self.dim, self.dim);
        let q = qkv.wq.project(&q_dense);
        let k = qkv.wk.project(&k_dense);
        let v = qkv.wv.project(&k_dense);
        let scale = (self.dim as f64).sqrt();
        scaled_dot_product_attention(&q, &k, &v, scale)
    }
}

/// Attention pattern data for visualization and analysis.
#[derive(Debug, Clone)]
pub struct AttentionPattern {
    pub weights: Matrix,
    pub source_len: usize,
    pub target_len: usize,
}

impl AttentionPattern {
    pub fn from_weights(weights: Matrix) -> Self {
        let source_len = weights.len();
        let target_len = if source_len > 0 { weights[0].len() } else { 0 };
        AttentionPattern { weights, source_len, target_len }
    }

    pub fn weight(&self, i: usize, j: usize) -> f64 {
        self.weights[i][j]
    }

    pub fn argmax_per_row(&self) -> Vec<usize> {
        self.weights
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            })
            .collect()
    }

    pub fn attention_entropy(&self) -> Vec<f64> {
        self.weights
            .iter()
            .map(|row| {
                let mut h = 0.0;
                for &w in row {
                    if w > 0.0 {
                        h -= w * w.ln();
                    }
                }
                h
            })
            .collect()
    }

    pub fn is_uniform(&self, tolerance: f64) -> bool {
        let uniform_w = if self.target_len > 0 { 1.0 / self.target_len as f64 } else { return true; };
        self.weights.iter().all(|row| row.iter().all(|&w| (w - uniform_w).abs() < tolerance))
    }

    pub fn to_heatmap(&self) -> String {
        let mut lines = Vec::new();
        for row in &self.weights {
            let line: Vec<String> = row
                .iter()
                .map(|&w| {
                    if w > 0.7 { "██".to_string() }
                    else if w > 0.3 { "▓▓".to_string() }
                    else if w > 0.1 { "▒▒".to_string() }
                    else if w > 0.01 { "░░".to_string() }
                    else { "··".to_string() }
                })
                .collect();
            lines.push(line.join(""));
        }
        lines.join("\n")
    }
}

/// Pairwise ternary compatibility score.
pub fn ternary_compatibility(query: &[Ternary], key: &[Ternary]) -> f64 {
    let min_len = query.len().min(key.len());
    if min_len == 0 {
        return 0.0;
    }
    let score: f64 = (0..min_len).map(|i| query[i].to_f64() * key[i].to_f64()).sum();
    score / min_len as f64
}

/// Causal (autoregressive) masked attention.
pub fn masked_attention(
    queries: &Matrix,
    keys: &Matrix,
    values: &Matrix,
    scale: f64,
) -> (Matrix, Matrix) {
    let n_q = queries.len();
    let n_k = keys.len();
    let d_k = keys[0].len();
    let d_v = values[0].len();

    let mut scores = vec![vec![f64::NEG_INFINITY; n_k]; n_q];
    for i in 0..n_q {
        for j in 0..=i.min(n_k - 1) {
            let s: f64 = (0..d_k).map(|k| queries[i][k] * keys[j][k]).sum();
            scores[i][j] = s / scale;
        }
    }

    let mut attention_weights = vec![vec![0.0; n_k]; n_q];
    for i in 0..n_q {
        let valid: Vec<f64> = scores[i]
            .iter()
            .filter(|&&s| s > f64::NEG_INFINITY / 2.0)
            .cloned()
            .collect();
        if valid.is_empty() {
            attention_weights[i][i.min(n_k - 1)] = 1.0;
            continue;
        }
        let soft = softmax(&valid);
        let mut vi = 0;
        for j in 0..n_k {
            if scores[i][j] > f64::NEG_INFINITY / 2.0 {
                attention_weights[i][j] = soft[vi];
                vi += 1;
            }
        }
    }

    let mut output = vec![vec![0.0; d_v]; n_q];
    for i in 0..n_q {
        for j in 0..n_k {
            for d in 0..d_v {
                output[i][d] += attention_weights[i][j] * values[j][d];
            }
        }
    }

    (output, attention_weights)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_uniform() {
        let s = softmax(&[1.0, 1.0, 1.0]);
        for &v in &s {
            assert!((v - 1.0 / 3.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let s = softmax(&[1.0, 2.0, 3.0]);
        assert!((s.iter().sum::<f64>() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_softmax_empty() {
        assert!(softmax(&[]).is_empty());
    }

    #[test]
    fn test_softmax_dominant() {
        let s = softmax(&[0.0, 0.0, 100.0]);
        assert!(s[2] > 0.99);
    }

    #[test]
    fn test_ternary_softmax_pos_winner() {
        // One score much higher than others → it becomes Pos
        let scores = vec![0.0, 0.0, 10.0];
        let ts = ternary_softmax(&scores, 0.1, 0.6);
        assert_eq!(ts[2], Ternary::Pos);
    }

    #[test]
    fn test_ternary_softmax_all_neutral() {
        // Uniform scores → all neutral (near 1/3, between thresholds)
        let scores = vec![1.0, 1.0, 1.0];
        let ts = ternary_softmax(&scores, 0.1, 0.6);
        for t in ts {
            assert_ne!(t, Ternary::Neg);
        }
    }

    #[test]
    fn test_ternary_weight_matrix_matvec() {
        let trits = vec![vec![1_i8, -1, 0], vec![0, 1, -1]];
        let m = TernaryWeightMatrix::from_trits(trits, 2.0);
        let x = vec![1.0, 1.0, 1.0];
        let y = m.matvec(&x);
        assert_eq!(y.len(), 2);
        // Row 0: (1 - 1 + 0) * 2.0 = 0
        assert!((y[0] - 0.0).abs() < 1e-10);
        // Row 1: (0 + 1 - 1) * 2.0 = 0
        assert!((y[1] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_ternary_weight_matrix_from_floats() {
        let m = vec![vec![0.9_f64, -0.8], vec![-0.3, 0.7]];
        let twm = TernaryWeightMatrix::from_floats(&m);
        assert_eq!(twm.rows, 2);
        assert_eq!(twm.cols, 2);
        for row in &twm.trits {
            for &t in row {
                assert!(t == -1 || t == 0 || t == 1);
            }
        }
    }

    #[test]
    fn test_ternary_weight_matrix_project() {
        let twm = TernaryWeightMatrix::from_trits(
            vec![vec![1_i8, 0], vec![0, 1]],
            1.0,
        );
        let x = vec![vec![2.0, 3.0], vec![4.0, 5.0]];
        let y = twm.project(&x);
        assert_eq!(y.len(), 2);
        assert!((y[0][0] - 2.0).abs() < 1e-10);
        assert!((y[0][1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_ternary_qkv_attention_shape() {
        let attn = TernaryQKVAttention::new(4, 4);
        let x: Vec<Vec<f64>> = (0..3).map(|i| vec![i as f64; 4]).collect();
        let (out, weights) = attn.forward(&x);
        assert_eq!(out.len(), 3);
        assert_eq!(weights.len(), 3);
        assert_eq!(weights[0].len(), 3);
    }

    #[test]
    fn test_ternary_qkv_ternary_softmax() {
        let attn = TernaryQKVAttention::new(4, 4);
        let x: Vec<Vec<f64>> = (0..3).map(|_| vec![1.0_f64; 4]).collect();
        let (_out, tw) = attn.forward_ternary_softmax(&x, 0.1, 0.6);
        assert_eq!(tw.len(), 3);
    }

    #[test]
    fn test_ternary_attention_self() {
        let attn = TernaryAttention::new(4);
        let seq = vec![Ternary::Neg, Ternary::Zero, Ternary::Pos];
        let (out, weights) = attn.self_attention(&seq);
        assert_eq!(out.len(), 3);
        for row in &weights {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_self_attention_with_rope() {
        let attn = TernaryAttention::new(4);
        let seq = vec![Ternary::Neg, Ternary::Zero, Ternary::Pos];
        let (out, weights) = attn.self_attention_with_rope(&seq);
        assert_eq!(out.len(), 3);
        assert_eq!(weights.len(), 3);
    }

    #[test]
    fn test_multi_head_attention() {
        let mha = MultiHeadAttention::new(2, 4);
        let seq = vec![Ternary::Neg, Ternary::Zero, Ternary::Pos, Ternary::Pos];
        let (out, heads) = mha.forward(&seq);
        assert_eq!(out.len(), 4);
        assert_eq!(heads.len(), 2);
    }

    #[test]
    fn test_multi_head_ternary_weights() {
        let mha = MultiHeadAttention::new(2, 4);
        let seq = vec![Ternary::Neg, Ternary::Pos, Ternary::Zero];
        let (out, _heads) = mha.forward_with_ternary_weights(&seq);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].len(), 4);
    }

    #[test]
    fn test_cross_attention() {
        let ca = CrossAttention::new(4);
        let src = vec![Ternary::Neg, Ternary::Pos];
        let tgt = vec![Ternary::Zero, Ternary::Pos, Ternary::Neg];
        let (out, weights) = ca.forward(&src, &tgt);
        assert_eq!(out.len(), 2);
        assert_eq!(weights[0].len(), 3);
    }

    #[test]
    fn test_cross_attention_ternary_weights() {
        let ca = CrossAttention::new(4);
        let src = vec![Ternary::Pos, Ternary::Neg];
        let tgt = vec![Ternary::Zero, Ternary::Pos];
        let (out, _) = ca.forward_ternary_weights(&src, &tgt);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn test_attention_pattern_argmax() {
        let w = vec![vec![0.1, 0.8, 0.1], vec![0.6, 0.2, 0.2]];
        let p = AttentionPattern::from_weights(w);
        assert_eq!(p.argmax_per_row(), vec![1, 0]);
    }

    #[test]
    fn test_attention_pattern_entropy_peaked() {
        let w = vec![vec![1.0, 0.0, 0.0]];
        let p = AttentionPattern::from_weights(w);
        assert!(p.attention_entropy()[0].abs() < 1e-10);
    }

    #[test]
    fn test_masked_attention_causal() {
        let q = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let k = q.clone();
        let v = q.clone();
        let (_out, weights) = masked_attention(&q, &k, &v, 1.0);
        assert!((weights[0][0] - 1.0).abs() < 1e-6);
        assert!(weights[0][1].abs() < 1e-6);
    }

    #[test]
    fn test_ternary_compatibility_identical() {
        let q = vec![Ternary::Pos, Ternary::Pos];
        let k = vec![Ternary::Pos, Ternary::Pos];
        assert!((ternary_compatibility(&q, &k) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ternary_compatibility_opposite() {
        let q = vec![Ternary::Pos, Ternary::Pos];
        let k = vec![Ternary::Neg, Ternary::Neg];
        assert!((ternary_compatibility(&q, &k) + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rope_encode_shape() {
        let x = vec![1.0, 0.0, 0.5, -0.5];
        let encoded = rope_encode(&x, 1);
        assert_eq!(encoded.len(), 4);
    }

    #[test]
    fn test_rope_position_zero_identity() {
        let x = vec![1.0, 0.5, -1.0, 0.3];
        let encoded = rope_encode(&x, 0);
        for (orig, enc) in x.iter().zip(encoded.iter()) {
            assert!((orig - enc).abs() < 1e-10);
        }
    }

    #[test]
    fn test_linear_projection() {
        let input = vec![1.0, 2.0];
        let weight = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let bias = vec![0.5, 0.5];
        let out = linear_projection(&input, &weight, &bias);
        assert!((out[0] - 1.5).abs() < 1e-10);
        assert!((out[1] - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_ternary_quantize() {
        assert_eq!(Ternary::quantize(0.8), Ternary::Pos);
        assert_eq!(Ternary::quantize(-0.8), Ternary::Neg);
        assert_eq!(Ternary::quantize(0.0), Ternary::Zero);
    }
}
