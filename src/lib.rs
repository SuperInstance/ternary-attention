#![forbid(unsafe_code)]

//! Attention mechanisms adapted for ternary inputs on {-1, 0, +1}.
//!
//! Provides TernaryAttention with softmax, multi-head attention, cross-attention
//! between ternary sequences, and attention pattern visualization as data.



/// A ternary value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Ternary {
    Neg,
    Zero,
    Pos,
}

impl Ternary {
    pub fn to_f64(self) -> f64 {
        match self {
            Ternary::Neg => -1.0,
            Ternary::Zero => 0.0,
            Ternary::Pos => 1.0,
        }
    }

    pub fn from_i8(v: i8) -> Option<Self> {
        match v {
            -1 => Some(Ternary::Neg),
            0 => Some(Ternary::Zero),
            1 => Some(Ternary::Pos),
            _ => None,
        }
    }
}

/// Softmax over a slice of f64 values (numerically stable).
pub fn softmax(scores: &[f64]) -> Vec<f64> {
    if scores.is_empty() {
        return vec![];
    }
    let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scores.iter().map(|&s| (s - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Dot product of two vectors.
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Matrix represented as row-major Vec<Vec<f64>>.
pub type Matrix = Vec<Vec<f64>>;

/// Create an identity matrix of size n.
pub fn identity(n: usize) -> Matrix {
    let mut m = vec![vec![0.0; n]; n];
    for i in 0..n {
        m[i][i] = 1.0;
    }
    m
}

/// Matrix multiply (a: k×m, b: m×n) -> k×n.
pub fn matmul(a: &Matrix, b: &Matrix) -> Matrix {
    let k = a.len();
    let m = a[0].len();
    let n = b[0].len();
    let mut result = vec![vec![0.0; n]; k];
    for i in 0..k {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..m {
                sum += a[i][l] * b[l][j];
            }
            result[i][j] = sum;
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

/// Scaled dot-product attention.
pub fn scaled_dot_product_attention(
    queries: &Matrix,
    keys: &Matrix,
    values: &Matrix,
    scale: f64,
) -> (Matrix, Matrix) {
    let n_q = queries.len();
    let n_k = keys.len();
    let d_k = keys[0].len();

    // Compute attention scores
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

    // Apply softmax row-wise
    let mut attention_weights = vec![vec![0.0; n_k]; n_q];
    for i in 0..n_q {
        let row: Vec<f64> = scores[i].clone();
        let soft = softmax(&row);
        attention_weights[i] = soft;
    }

    // Weighted sum of values
    let d_v = values[0].len();
    let mut output = vec![vec![0.0; d_v]; n_q];
    for i in 0..n_q {
        for j in 0..d_v {
            let mut sum = 0.0;
            for k in 0..n_k {
                sum += attention_weights[i][k] * values[k][j];
            }
            output[i][j] = sum;
        }
    }

    (output, attention_weights)
}

/// Convert a ternary sequence to dense vectors.
pub fn ternary_to_dense(sequence: &[Ternary], dim: usize) -> Matrix {
    sequence
        .iter()
        .map(|&t| {
            let base = t.to_f64();
            (0..dim).map(|i| base * (i as f64 + 1.0) / dim as f64).collect()
        })
        .collect()
}

/// Ternary attention: attention over ternary-valued sequences.
pub struct TernaryAttention {
    pub dim: usize,
    pub scale: f64,
}

impl TernaryAttention {
    pub fn new(dim: usize) -> Self {
        TernaryAttention {
            dim,
            scale: (dim as f64).sqrt(),
        }
    }

    /// Compute self-attention on a ternary sequence.
    pub fn self_attention(&self, sequence: &[Ternary]) -> (Matrix, Matrix) {
        let dense = ternary_to_dense(sequence, self.dim);
        scaled_dot_product_attention(&dense, &dense, &dense, self.scale)
    }

    /// Compute attention with explicit Q, K, V from ternary sequences.
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
}

/// Multi-head attention for ternary inputs.
pub struct MultiHeadAttention {
    pub n_heads: usize,
    pub dim: usize,
    pub head_dim: usize,
}

impl MultiHeadAttention {
    pub fn new(n_heads: usize, dim: usize) -> Self {
        MultiHeadAttention {
            n_heads,
            dim,
            head_dim: dim / n_heads,
        }
    }

    /// Compute multi-head self-attention on ternary sequence.
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
}

/// Cross-attention between two ternary sequences.
pub struct CrossAttention {
    pub dim: usize,
}

impl CrossAttention {
    pub fn new(dim: usize) -> Self {
        CrossAttention { dim }
    }

    /// Cross-attention: queries from source, keys/values from target.
    pub fn forward(&self, source: &[Ternary], target: &[Ternary]) -> (Matrix, Matrix) {
        let q = ternary_to_dense(source, self.dim);
        let k = ternary_to_dense(target, self.dim);
        let v = ternary_to_dense(target, self.dim);
        let scale = (self.dim as f64).sqrt();
        scaled_dot_product_attention(&q, &k, &v, scale)
    }
}

/// Attention pattern as data for visualization.
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
        AttentionPattern {
            weights,
            source_len,
            target_len,
        }
    }

    /// Get the attention weight from source position i to target position j.
    pub fn weight(&self, i: usize, j: usize) -> f64 {
        self.weights[i][j]
    }

    /// Get the most attended position for each source.
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

    /// Compute the entropy of attention distribution for each source position.
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

    /// Check if attention is approximately uniform.
    pub fn is_uniform(&self, tolerance: f64) -> bool {
        let uniform_w = if self.target_len > 0 {
            1.0 / self.target_len as f64
        } else {
            return true;
        };
        self.weights.iter().all(|row| {
            row.iter()
                .all(|&w| (w - uniform_w).abs() < tolerance)
        })
    }

    /// Convert to a flat heatmap string for debugging.
    pub fn to_heatmap(&self) -> String {
        let mut lines = Vec::new();
        for row in &self.weights {
            let line: Vec<String> = row
                .iter()
                .map(|&w| {
                    if w > 0.7 {
                        "██".to_string()
                    } else if w > 0.3 {
                        "▓▓".to_string()
                    } else if w > 0.1 {
                        "▒▒".to_string()
                    } else if w > 0.01 {
                        "░░".to_string()
                    } else {
                        "··".to_string()
                    }
                })
                .collect();
            lines.push(line.join(""));
        }
        lines.join("\n")
    }
}

/// Compute attention between ternary sequences using a simple compatibility function.
pub fn ternary_compatibility(query: &[Ternary], key: &[Ternary]) -> f64 {
    let min_len = query.len().min(key.len());
    let mut score = 0.0;
    for i in 0..min_len {
        let q = query[i].to_f64();
        let k = key[i].to_f64();
        // Ternary compatibility: +1 for match, -1 for opposite, 0 for neutral
        score += q * k;
    }
    score / min_len as f64
}

/// Masked attention: apply causal mask to scores.
pub fn masked_attention(
    queries: &Matrix,
    keys: &Matrix,
    values: &Matrix,
    scale: f64,
) -> (Matrix, Matrix) {
    let n_q = queries.len();
    let n_k = keys.len();
    let d_k = keys[0].len();

    let mut scores = vec![vec![f64::NEG_INFINITY; n_k]; n_q];
    for i in 0..n_q {
        for j in 0..n_k {
            if j <= i {
                let mut s = 0.0;
                for k in 0..d_k {
                    s += queries[i][k] * keys[j][k];
                }
                scores[i][j] = s / scale;
            }
        }
    }

    let mut attention_weights = vec![vec![0.0; n_k]; n_q];
    for i in 0..n_q {
        let valid: Vec<f64> = scores[i].iter().filter(|&&s| s > f64::NEG_INFINITY / 2.0).cloned().collect();
        if valid.is_empty() {
            attention_weights[i][i.min(n_k - 1)] = 1.0;
            continue;
        }
        let soft = softmax(&valid);
        let mut idx = 0;
        for j in 0..n_k {
            if scores[i][j] > f64::NEG_INFINITY / 2.0 {
                attention_weights[i][j] = soft[idx];
                idx += 1;
            }
        }
    }

    let d_v = values[0].len();
    let mut output = vec![vec![0.0; d_v]; n_q];
    for i in 0..n_q {
        for j in 0..d_v {
            for k in 0..n_k {
                output[i][j] += attention_weights[i][k] * values[k][j];
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
        let sum: f64 = s.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_softmax_empty() {
        let s = softmax(&[]);
        assert!(s.is_empty());
    }

    #[test]
    fn test_softmax_dominant() {
        let s = softmax(&[0.0, 0.0, 100.0]);
        assert!(s[2] > 0.99);
    }

    #[test]
    fn test_dot_product() {
        let d = dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        assert!((d - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_identity_matrix() {
        let i = identity(3);
        assert_eq!(i[0], vec![1.0, 0.0, 0.0]);
        assert_eq!(i[1], vec![0.0, 1.0, 0.0]);
        assert_eq!(i[2], vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_matmul_identity() {
        let i = identity(2);
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let result = matmul(&a, &i);
        assert!((result[0][0] - 1.0).abs() < 1e-10);
        assert!((result[0][1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_ternary_attention_self() {
        let attn = TernaryAttention::new(4);
        let seq = vec![Ternary::Neg, Ternary::Zero, Ternary::Pos];
        let (output, weights) = attn.self_attention(&seq);
        assert_eq!(output.len(), 3);
        assert_eq!(weights.len(), 3);
        // Each row of weights should sum to ~1
        for row in &weights {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_multi_head_attention() {
        let mha = MultiHeadAttention::new(2, 4);
        let seq = vec![Ternary::Neg, Ternary::Zero, Ternary::Pos, Ternary::Pos];
        let (output, head_weights) = mha.forward(&seq);
        assert_eq!(output.len(), 4);
        assert_eq!(output[0].len(), 4);
        assert_eq!(head_weights.len(), 2);
    }

    #[test]
    fn test_cross_attention() {
        let ca = CrossAttention::new(4);
        let source = vec![Ternary::Neg, Ternary::Pos];
        let target = vec![Ternary::Zero, Ternary::Pos, Ternary::Neg];
        let (output, weights) = ca.forward(&source, &target);
        assert_eq!(output.len(), 2);
        assert_eq!(weights[0].len(), 3);
    }

    #[test]
    fn test_attention_pattern_from_weights() {
        let w = vec![vec![0.5, 0.3, 0.2], vec![0.1, 0.8, 0.1]];
        let p = AttentionPattern::from_weights(w);
        assert_eq!(p.source_len, 2);
        assert_eq!(p.target_len, 3);
        assert!((p.weight(0, 0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_attention_pattern_argmax() {
        let w = vec![vec![0.1, 0.8, 0.1], vec![0.6, 0.2, 0.2]];
        let p = AttentionPattern::from_weights(w);
        assert_eq!(p.argmax_per_row(), vec![1, 0]);
    }

    #[test]
    fn test_attention_entropy() {
        let w = vec![vec![1.0, 0.0, 0.0]];
        let p = AttentionPattern::from_weights(w);
        let h = p.attention_entropy();
        assert!(h[0].abs() < 1e-10);
    }

    #[test]
    fn test_attention_pattern_heatmap() {
        let w = vec![vec![0.8, 0.1], vec![0.2, 0.7]];
        let p = AttentionPattern::from_weights(w);
        let heatmap = p.to_heatmap();
        assert!(!heatmap.is_empty());
    }

    #[test]
    fn test_ternary_compatibility() {
        let q = vec![Ternary::Pos, Ternary::Pos];
        let k = vec![Ternary::Pos, Ternary::Pos];
        let score = ternary_compatibility(&q, &k);
        assert!((score - 1.0).abs() < 1e-10);

        let k2 = vec![Ternary::Neg, Ternary::Neg];
        let score2 = ternary_compatibility(&q, &k2);
        assert!((score2 - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_masked_attention() {
        let q = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let k = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let v = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let (_output, weights) = masked_attention(&q, &k, &v, 1.0);
        // Position 0 should only attend to position 0
        assert!((weights[0][0] - 1.0).abs() < 1e-6);
        assert!((weights[0][1]).abs() < 1e-6);
    }

    #[test]
    fn test_ternary_to_dense() {
        let seq = vec![Ternary::Pos];
        let dense = ternary_to_dense(&seq, 3);
        assert_eq!(dense.len(), 1);
        assert_eq!(dense[0].len(), 3);
        assert!(dense[0][0] > 0.0);
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
    fn test_ternary_from_i8() {
        assert_eq!(Ternary::from_i8(-1), Some(Ternary::Neg));
        assert_eq!(Ternary::from_i8(0), Some(Ternary::Zero));
        assert_eq!(Ternary::from_i8(1), Some(Ternary::Pos));
        assert_eq!(Ternary::from_i8(5), None);
    }

    #[test]
    fn test_attention_pattern_uniform() {
        let w = vec![vec![0.333, 0.333, 0.334], vec![0.333, 0.333, 0.334]];
        let p = AttentionPattern::from_weights(w);
        assert!(p.is_uniform(0.01));
    }

    #[test]
    fn test_cross_attention_different_lengths() {
        let ca = CrossAttention::new(4);
        let source = vec![Ternary::Neg];
        let target = vec![Ternary::Zero, Ternary::Pos, Ternary::Neg, Ternary::Pos];
        let (output, weights) = ca.forward(&source, &target);
        assert_eq!(output.len(), 1);
        assert_eq!(weights[0].len(), 4);
    }
}
