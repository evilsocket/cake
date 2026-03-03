//! Causal self attention implementation with fused QKV projection.
use candle_core::{DType, Result, Tensor, D};
use candle_nn::{linear_no_bias, Linear, Module, VarBuilder};

#[derive(Debug, Clone)]
pub struct CausalSelfAttention {
    qkv_proj: Linear,
    o_proj: Linear,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    size_q: usize,
    size_kv: usize,
}

#[inline]
fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?
        .to_dtype(on_false.dtype())?
        .broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

impl CausalSelfAttention {
    fn apply_rotary_emb(
        &self,
        x: &Tensor,
        index_pos: usize,
        cache: &super::Cache,
    ) -> Result<Tensor> {
        let (_batch_size, _, seq_len, _hidden_size) = x.dims4()?;
        let cos = cache.cosine(index_pos, seq_len, &x.device())?;
        let sin = cache.sine(index_pos, seq_len, &x.device())?;
        candle_nn::rotary_emb::rope(x, &cos, &sin)
    }

    /// Process the input tensor using the given state indexes and cache.
    pub fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut super::Cache,
    ) -> anyhow::Result<Tensor> {
        let (b_sz, seq_len, hidden_size) = x.dims3().map_err(|e| anyhow!("x.dims3 -> {e}"))?;

        // Single fused QKV projection
        let qkv = self
            .qkv_proj
            .forward(x)
            .map_err(|e| anyhow!("qkv.forward -> {e}"))?;

        let q = qkv
            .narrow(D::Minus1, 0, self.size_q)
            .map_err(|e| anyhow!("q split -> {e}"))?;
        let k = qkv
            .narrow(D::Minus1, self.size_q, self.size_kv)
            .map_err(|e| anyhow!("k split -> {e}"))?;
        let v = qkv
            .narrow(D::Minus1, self.size_q + self.size_kv, self.size_kv)
            .map_err(|e| anyhow!("v split -> {e}"))?;

        let q = q
            .reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()
            .map_err(|e| anyhow!("q.reshape -> {e}"))?;
        let k = k
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()
            .map_err(|e| anyhow!("k.reshape -> {e}"))?;
        let v = v
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)
            .map_err(|e| anyhow!("v.reshape -> {e}"))?;

        let q = self
            .apply_rotary_emb(&q, index_pos, cache)
            .map_err(|e| anyhow!("q.apply_rotary_emb -> {e}"))?;

        let k = self
            .apply_rotary_emb(&k, index_pos, cache)
            .map_err(|e| anyhow!("k.apply_rotary_emb -> {e}"))?;

        let (k, v) = cache
            .process_kv(block_idx, k, v)
            .map_err(|e| anyhow!("cache.process_kv(block={block_idx}) -> {e}"))?;

        // Compute attention in F32 for numerical stability.
        let in_dtype = q.dtype();
        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;

        #[allow(unused_labels)]
        let y = 'attn: {
            // Fused SDPA on Metal — single kernel, native GQA (no repeat_kv needed)
            #[cfg(feature = "metal")]
            if matches!(q.device(), candle_core::Device::Metal(_)) {
                let scale = 1.0 / (self.head_dim as f32).sqrt();
                break 'attn candle_nn::ops::sdpa(&q, &k, &v, None, seq_len > 1, scale, 1.0)
                    .map_err(|e| anyhow!("sdpa: {e}"))?;
            }

            // Manual attention with GQA head expansion (CUDA, CPU)
            let k = self
                .repeat_kv(k)
                .map_err(|e| anyhow!("repeat_kv(k) -> {e}"))?;
            let v = self
                .repeat_kv(v)
                .map_err(|e| anyhow!("repeat_kv(v) -> {e}"))?;

            let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
            let att = if seq_len == 1 {
                att
            } else {
                let mask = cache
                    .mask(seq_len, &q.device())
                    .map_err(|e| anyhow!("cache.mask({seq_len}) -> {e}"))?
                    .broadcast_as(att.shape())
                    .map_err(|e| anyhow!("mask.broadcast_as({:?}) -> {e}", att.shape()))?;

                masked_fill(&att, &mask, f32::NEG_INFINITY)
                    .map_err(|e| anyhow!("masked_fill -> {e}"))?
            };
            let att = candle_nn::ops::softmax_last_dim(&att)?;
            att.matmul(&v.contiguous()?)?
        };

        let y = y.to_dtype(in_dtype)?;
        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, hidden_size])?;
        let y = self.o_proj.forward(&y)?;

        Ok(y)
    }

    fn repeat_kv(&self, x: Tensor) -> Result<Tensor> {
        candle_transformers::utils::repeat_kv(
            x,
            self.num_attention_heads / self.num_key_value_heads,
        )
    }

    /// Load an instance of this object from the VarBuilder object with the given configuration.
    pub fn load(vb: VarBuilder, cfg: &super::Config) -> Result<Self> {
        let size_in = cfg.hidden_size;
        let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
        let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;

        // Fuse Q, K, V projections into a single Linear
        let (qkv_proj,) = if cfg.use_qkv_bias {
            let q_w = vb.pp("q_proj").get((size_q, size_in), "weight")?;
            let k_w = vb.pp("k_proj").get((size_kv, size_in), "weight")?;
            let v_w = vb.pp("v_proj").get((size_kv, size_in), "weight")?;
            let fused_w = Tensor::cat(&[&q_w, &k_w, &v_w], 0)?;

            let q_b = vb.pp("q_proj").get(size_q, "bias")?;
            let k_b = vb.pp("k_proj").get(size_kv, "bias")?;
            let v_b = vb.pp("v_proj").get(size_kv, "bias")?;
            let fused_b = Tensor::cat(&[&q_b, &k_b, &v_b], 0)?;

            (Linear::new(fused_w, Some(fused_b)),)
        } else {
            let q_w = vb.pp("q_proj").get((size_q, size_in), "weight")?;
            let k_w = vb.pp("k_proj").get((size_kv, size_in), "weight")?;
            let v_w = vb.pp("v_proj").get((size_kv, size_in), "weight")?;
            let fused_w = Tensor::cat(&[&q_w, &k_w, &v_w], 0)?;

            (Linear::new(fused_w, None),)
        };

        // o_proj never has bias in either architecture
        let o_proj = linear_no_bias(size_q, size_in, vb.pp("o_proj"))?;

        Ok(Self {
            qkv_proj,
            o_proj,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: cfg.hidden_size / cfg.num_attention_heads,
            size_q,
            size_kv,
        })
    }
}
