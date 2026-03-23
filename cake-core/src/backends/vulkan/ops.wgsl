// ─── WGSL compute shaders for Vulkan backend ───────────────────────

// ═══════════════════════════════════════════════════════════════════
// Elementwise ops (2-input): binding 0=A, 1=B, 2=output, 3=params
// ═══════════════════════════════════════════════════════════════════

@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

struct Params { count: u32, }

@compute @workgroup_size(256)
fn silu_mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Process 4 elements per thread via vec4 for better throughput
    let base = gid.x * 4u;
    let count = params.count;
    if (base >= count) { return; }

    // Fast path: full vec4
    if (base + 3u < count) {
        let g = vec4(input_a[base], input_a[base + 1u], input_a[base + 2u], input_a[base + 3u]);
        let b = vec4(input_b[base], input_b[base + 1u], input_b[base + 2u], input_b[base + 3u]);
        let s = g / (vec4(1.0) + exp(-g)) * b;
        output[base] = s.x; output[base + 1u] = s.y;
        output[base + 2u] = s.z; output[base + 3u] = s.w;
    } else {
        // Tail: process remaining elements
        for (var j = 0u; j < 4u; j++) {
            let idx = base + j;
            if (idx < count) {
                let g = input_a[idx];
                output[idx] = g / (1.0 + exp(-g)) * input_b[idx];
            }
        }
    }
}

@compute @workgroup_size(256)
fn exp_mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let base = gid.x * 4u;
    let count = params.count;
    if (base >= count) { return; }
    if (base + 3u < count) {
        let a = vec4(input_a[base], input_a[base + 1u], input_a[base + 2u], input_a[base + 3u]);
        let b = vec4(input_b[base], input_b[base + 1u], input_b[base + 2u], input_b[base + 3u]);
        let s = a * exp(b);
        output[base] = s.x; output[base + 1u] = s.y;
        output[base + 2u] = s.z; output[base + 3u] = s.w;
    } else {
        for (var j = 0u; j < 4u; j++) {
            let idx = base + j;
            if (idx < count) { output[idx] = input_a[idx] * exp(input_b[idx]); }
        }
    }
}

@compute @workgroup_size(256)
fn stable_softplus(@builtin(global_invocation_id) gid: vec3<u32>) {
    let base = gid.x * 4u;
    let count = params.count;
    if (base >= count) { return; }
    if (base + 3u < count) {
        let x = vec4(input_a[base], input_a[base + 1u], input_a[base + 2u], input_a[base + 3u]);
        let s = max(x, log(exp(min(x, vec4(88.0))) + vec4(1.0)));
        output[base] = s.x; output[base + 1u] = s.y;
        output[base + 2u] = s.z; output[base + 3u] = s.w;
    } else {
        for (var j = 0u; j < 4u; j++) {
            let idx = base + j;
            if (idx < count) {
                let x = input_a[idx];
                output[idx] = max(x, log(exp(min(x, 88.0)) + 1.0));
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// 3-input ops: extra binding 4=C
// ═══════════════════════════════════════════════════════════════════

@group(0) @binding(4) var<storage, read> input_c: array<f32>;

@compute @workgroup_size(256)
fn sub_mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let base = gid.x * 4u;
    let count = params.count;
    if (base >= count) { return; }
    if (base + 3u < count) {
        let a = vec4(input_a[base], input_a[base + 1u], input_a[base + 2u], input_a[base + 3u]);
        let b = vec4(input_b[base], input_b[base + 1u], input_b[base + 2u], input_b[base + 3u]);
        let c = vec4(input_c[base], input_c[base + 1u], input_c[base + 2u], input_c[base + 3u]);
        let s = (a - b) * c;
        output[base] = s.x; output[base + 1u] = s.y;
        output[base + 2u] = s.z; output[base + 3u] = s.w;
    } else {
        for (var j = 0u; j < 4u; j++) {
            let idx = base + j;
            if (idx < count) { output[idx] = (input_a[idx] - input_b[idx]) * input_c[idx]; }
        }
    }
}

@compute @workgroup_size(256)
fn add3(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Process 4 elements per thread via vec4
    let base = gid.x * 4u;
    let count = params.count;
    if (base >= count) { return; }

    if (base + 3u < count) {
        let a = vec4(input_a[base], input_a[base + 1u], input_a[base + 2u], input_a[base + 3u]);
        let b = vec4(input_b[base], input_b[base + 1u], input_b[base + 2u], input_b[base + 3u]);
        let c = vec4(input_c[base], input_c[base + 1u], input_c[base + 2u], input_c[base + 3u]);
        let s = a + b + c;
        output[base] = s.x; output[base + 1u] = s.y;
        output[base + 2u] = s.z; output[base + 3u] = s.w;
    } else {
        for (var j = 0u; j < 4u; j++) {
            let idx = base + j;
            if (idx < count) {
                output[idx] = input_a[idx] + input_b[idx] + input_c[idx];
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// GEMV: y[j] = sum_i(x[i] * W[i * N + j])  for x[1,K] × W[K,N] = y[1,N]
//
// One thread per output column. Adjacent threads read adjacent weight
// columns (coalesced). x is tiled into shared memory (broadcast to all).
// No reduction needed — each thread accumulates its own dot product.
// ═══════════════════════════════════════════════════════════════════

struct GemvParams {
    N: u32,
    K: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> gemv_x: array<f32>;    // [K]
@group(0) @binding(1) var<storage, read> gemv_w: array<f32>;    // [K, N] row-major
@group(0) @binding(2) var<storage, read_write> gemv_y: array<f32>; // [N]
@group(0) @binding(3) var<uniform> gemv_params: GemvParams;

var<workgroup> gemv_shared: array<f32, 256>;

@compute @workgroup_size(256)
fn gemv(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let col = gid.x;
    let tid = lid.x;
    let N = gemv_params.N;
    let K = gemv_params.K;

    var acc: f32 = 0.0;
    let num_tiles = (K + 255u) / 256u;

    for (var t: u32 = 0u; t < num_tiles; t++) {
        // Cooperatively load 256 elements of x into shared memory
        let x_idx = t * 256u + tid;
        if (x_idx < K) {
            gemv_shared[tid] = gemv_x[x_idx];
        } else {
            gemv_shared[tid] = 0.0;
        }
        workgroupBarrier();

        // Each thread accumulates dot product for its output column.
        // Adjacent threads read adjacent weight columns — coalesced access.
        if (col < N) {
            let tile_base = t * 256u;
            let tile_end = min(256u, K - tile_base);
            for (var k: u32 = 0u; k < tile_end; k++) {
                acc += gemv_shared[k] * gemv_w[(tile_base + k) * N + col];
            }
        }
        workgroupBarrier();
    }

    if (col < N) {
        gemv_y[col] = acc;
    }
}

// ═══════════════════════════════════════════════════════════════════
// GEMV F16: same as gemv but weights stored as packed F16 (u32 pairs).
// Halves bandwidth vs F32 weights. unpack2x16float converts in-register.
// ═══════════════════════════════════════════════════════════════════

@group(0) @binding(0) var<storage, read> gemvh_x: array<f32>;    // [K] F32
@group(0) @binding(1) var<storage, read> gemvh_w: array<u32>;    // [K, N/2] packed F16
@group(0) @binding(2) var<storage, read_write> gemvh_y: array<f32>; // [N] F32
@group(0) @binding(3) var<uniform> gemvh_params: GemvParams;

var<workgroup> gemvh_shared: array<f32, 256>;

@compute @workgroup_size(256)
fn gemv_f16(@builtin(global_invocation_id) gid: vec3<u32>,
            @builtin(local_invocation_id) lid: vec3<u32>) {
    let col = gid.x;
    let tid = lid.x;
    let N = gemvh_params.N;
    let K = gemvh_params.K;
    let half_N = N / 2u;

    var acc: f32 = 0.0;
    let num_tiles = (K + 255u) / 256u;

    for (var t: u32 = 0u; t < num_tiles; t++) {
        let x_idx = t * 256u + tid;
        if (x_idx < K) {
            gemvh_shared[tid] = gemvh_x[x_idx];
        } else {
            gemvh_shared[tid] = 0.0;
        }
        workgroupBarrier();

        if (col < N) {
            let tile_base = t * 256u;
            let tile_end = min(256u, K - tile_base);
            let col_pair = col / 2u;
            let col_half = col % 2u;
            for (var k: u32 = 0u; k < tile_end; k++) {
                let packed = gemvh_w[(tile_base + k) * half_N + col_pair];
                let unpacked = unpack2x16float(packed);
                let w_val = select(unpacked.x, unpacked.y, col_half == 1u);
                acc += gemvh_shared[k] * w_val;
            }
        }
        workgroupBarrier();
    }

    if (col < N) {
        gemvh_y[col] = acc;
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tiled GEMM: C[M,N] = A[M,K] × B[K,N]  (for M > 1)
// 32×64 output tile with 2×4 register tiling, K-tile = 32.
// 16×16 workgroup (256 threads = 4 RDNA 2 wavefronts). Each thread
// computes a 2×4 block (8 MADs/k-step). Halves workgroup count in N.
// Shared memory: tile_a[32,33] + tile_b[32,65].
// ═══════════════════════════════════════════════════════════════════

struct MatmulParams {
    M: u32,
    N: u32,
    K: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> mat_a: array<f32>;
@group(0) @binding(1) var<storage, read> mat_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> mat_c: array<f32>;
@group(0) @binding(3) var<uniform> mat_params: MatmulParams;

const TILE_M: u32 = 32;   // output tile M dimension
const TILE_N: u32 = 64;   // output tile N dimension (2x wider)
const TILE_K: u32 = 32;   // K dimension tile
const TILE_A_STRIDE: u32 = 33;  // padded stride (32+1)
const TILE_B_STRIDE: u32 = 65;  // padded stride (64+1)
var<workgroup> tile_a: array<f32, 1056>;  // [32, 33] padded
var<workgroup> tile_b: array<f32, 2080>;  // [32, 65] padded

@compute @workgroup_size(16, 16)
fn matmul(@builtin(global_invocation_id) gid: vec3<u32>,
          @builtin(local_invocation_id) lid: vec3<u32>) {
    let lr = lid.x;
    let lc = lid.y;
    let M = mat_params.M;
    let N = mat_params.N;
    let K = mat_params.K;

    // Workgroup base position (32×64 output tile)
    let wg_row = gid.x * 2u - lid.x * 2u;
    let wg_col = gid.y * 4u - lid.y * 4u;
    let row0 = wg_row + lr * 2u;
    let col0 = wg_col + lc * 4u;

    // 2×4 accumulators (8 registers)
    var acc00: f32 = 0.0; var acc01: f32 = 0.0; var acc02: f32 = 0.0; var acc03: f32 = 0.0;
    var acc10: f32 = 0.0; var acc11: f32 = 0.0; var acc12: f32 = 0.0; var acc13: f32 = 0.0;
    let num_tiles = (K + TILE_K - 1u) / TILE_K;

    // Linear thread index
    let lin = lr * 16u + lc;

    for (var t: u32 = 0u; t < num_tiles; t++) {
        let tk = t * TILE_K;

        // Cooperative load tile_a[32, 32]: 256 threads, 4 elements each
        for (var i: u32 = 0u; i < 4u; i++) {
            let idx = lin * 4u + i;
            let tr = idx / TILE_K;
            let tc = idx % TILE_K;
            let a_row = wg_row + tr;
            let a_col = tk + tc;
            if (a_row < M && a_col < K) {
                tile_a[tr * TILE_A_STRIDE + tc] = mat_a[a_row * K + a_col];
            } else {
                tile_a[tr * TILE_A_STRIDE + tc] = 0.0;
            }
        }

        // Cooperative load tile_b[32, 64]: 256 threads, 8 elements each
        for (var i: u32 = 0u; i < 8u; i++) {
            let idx = lin * 8u + i;
            let tr = idx / TILE_N;
            let tc = idx % TILE_N;
            let b_row = tk + tr;
            let b_col = wg_col + tc;
            if (b_row < K && b_col < N) {
                tile_b[tr * TILE_B_STRIDE + tc] = mat_b[b_row * N + b_col];
            } else {
                tile_b[tr * TILE_B_STRIDE + tc] = 0.0;
            }
        }

        workgroupBarrier();

        let r0 = lr * 2u;
        let r1 = r0 + 1u;
        let c0 = lc * 4u;
        for (var k: u32 = 0u; k < TILE_K; k++) {
            let a0k = tile_a[r0 * TILE_A_STRIDE + k];
            let a1k = tile_a[r1 * TILE_A_STRIDE + k];
            let bk0 = tile_b[k * TILE_B_STRIDE + c0];
            let bk1 = tile_b[k * TILE_B_STRIDE + c0 + 1u];
            let bk2 = tile_b[k * TILE_B_STRIDE + c0 + 2u];
            let bk3 = tile_b[k * TILE_B_STRIDE + c0 + 3u];
            acc00 += a0k * bk0; acc01 += a0k * bk1; acc02 += a0k * bk2; acc03 += a0k * bk3;
            acc10 += a1k * bk0; acc11 += a1k * bk1; acc12 += a1k * bk2; acc13 += a1k * bk3;
        }
        workgroupBarrier();
    }

    // Write 2×4 output block
    if (row0 < M && col0 < N) { mat_c[row0 * N + col0] = acc00; }
    if (row0 < M && col0 + 1u < N) { mat_c[row0 * N + col0 + 1u] = acc01; }
    if (row0 < M && col0 + 2u < N) { mat_c[row0 * N + col0 + 2u] = acc02; }
    if (row0 < M && col0 + 3u < N) { mat_c[row0 * N + col0 + 3u] = acc03; }
    if (row0 + 1u < M && col0 < N) { mat_c[(row0 + 1u) * N + col0] = acc10; }
    if (row0 + 1u < M && col0 + 1u < N) { mat_c[(row0 + 1u) * N + col0 + 1u] = acc11; }
    if (row0 + 1u < M && col0 + 2u < N) { mat_c[(row0 + 1u) * N + col0 + 2u] = acc12; }
    if (row0 + 1u < M && col0 + 3u < N) { mat_c[(row0 + 1u) * N + col0 + 3u] = acc13; }
}

// ═══════════════════════════════════════════════════════════════════
// Scaled softmax: output[row,j] = softmax(input[row,j] * scale)
// with optional causal masking.
//
// One workgroup per row. 256 threads cooperate on the reduction.
// Three passes: (1) find max, (2) exp + sum, (3) normalize.
// Causal mask: if seq_len > 0, mask positions j > kv_len - seq_len + q_idx.
// ═══════════════════════════════════════════════════════════════════

struct SoftmaxParams {
    rows: u32,      // total rows (batch * heads * seq_len)
    cols: u32,      // row length (kv_len)
    scale_bits: u32, // f32 scale reinterpreted as u32
    seq_len: u32,   // 0 = no causal mask, >0 = causal with this seq_len
}

@group(0) @binding(0) var<storage, read> sm_in: array<f32>;
@group(0) @binding(1) var<storage, read> sm_dummy: array<f32>;  // unused, keeps 4-binding layout
@group(0) @binding(2) var<storage, read_write> sm_out: array<f32>;
@group(0) @binding(3) var<uniform> sm_params: SoftmaxParams;

var<workgroup> sm_shared: array<f32, 256>;

@compute @workgroup_size(256)
fn scaled_softmax(@builtin(workgroup_id) wid: vec3<u32>,
                  @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x;
    let tid = lid.x;
    let cols = sm_params.cols;
    let scale = bitcast<f32>(sm_params.scale_bits);
    let base = row * cols;

    // Causal mask boundary: positions j > max_j are masked to -inf
    var max_j = cols;  // no mask by default
    if (sm_params.seq_len > 0u) {
        let q_idx = row % sm_params.seq_len;
        max_j = cols - sm_params.seq_len + q_idx + 1u;
    }

    // Pass 1: find row maximum (for numerical stability)
    var local_max: f32 = -3.4e38;
    var i = tid;
    while (i < cols) {
        var val = sm_in[base + i] * scale;
        if (i >= max_j) { val = -3.4e38; }  // causal mask
        local_max = max(local_max, val);
        i += 256u;
    }
    sm_shared[tid] = local_max;
    workgroupBarrier();

    // Parallel max reduction
    if (tid < 128u) { sm_shared[tid] = max(sm_shared[tid], sm_shared[tid + 128u]); }
    workgroupBarrier();
    if (tid < 64u) { sm_shared[tid] = max(sm_shared[tid], sm_shared[tid + 64u]); }
    workgroupBarrier();
    if (tid < 32u) { sm_shared[tid] = max(sm_shared[tid], sm_shared[tid + 32u]); }
    workgroupBarrier();
    if (tid < 16u) { sm_shared[tid] = max(sm_shared[tid], sm_shared[tid + 16u]); }
    workgroupBarrier();
    if (tid < 8u) { sm_shared[tid] = max(sm_shared[tid], sm_shared[tid + 8u]); }
    workgroupBarrier();
    if (tid < 4u) { sm_shared[tid] = max(sm_shared[tid], sm_shared[tid + 4u]); }
    workgroupBarrier();
    if (tid < 2u) { sm_shared[tid] = max(sm_shared[tid], sm_shared[tid + 2u]); }
    workgroupBarrier();
    if (tid == 0u) { sm_shared[0] = max(sm_shared[0], sm_shared[1]); }
    workgroupBarrier();
    let row_max = sm_shared[0];

    // Pass 2: compute exp(val - max) and accumulate sum
    var local_sum: f32 = 0.0;
    i = tid;
    while (i < cols) {
        var val = sm_in[base + i] * scale;
        if (i >= max_j) { val = -3.4e38; }
        let e = exp(val - row_max);
        sm_out[base + i] = e;  // store exp values
        local_sum += e;
        i += 256u;
    }
    sm_shared[tid] = local_sum;
    workgroupBarrier();

    // Parallel sum reduction
    if (tid < 128u) { sm_shared[tid] += sm_shared[tid + 128u]; }
    workgroupBarrier();
    if (tid < 64u) { sm_shared[tid] += sm_shared[tid + 64u]; }
    workgroupBarrier();
    if (tid < 32u) { sm_shared[tid] += sm_shared[tid + 32u]; }
    workgroupBarrier();
    if (tid < 16u) { sm_shared[tid] += sm_shared[tid + 16u]; }
    workgroupBarrier();
    if (tid < 8u) { sm_shared[tid] += sm_shared[tid + 8u]; }
    workgroupBarrier();
    if (tid < 4u) { sm_shared[tid] += sm_shared[tid + 4u]; }
    workgroupBarrier();
    if (tid < 2u) { sm_shared[tid] += sm_shared[tid + 2u]; }
    workgroupBarrier();
    if (tid == 0u) { sm_shared[0] += sm_shared[1]; }
    workgroupBarrier();
    let inv_sum = 1.0 / sm_shared[0];

    // Pass 3: normalize by dividing by sum
    i = tid;
    while (i < cols) {
        sm_out[base + i] *= inv_sum;
        i += 256u;
    }
}

// ═══════════════════════════════════════════════════════════════════
// Small-M GEMM: C[M,N] = A[M,K] × B[K,N]  (for M <= 16)
// 16×64 output tile with 1×4 register tiling, K-tile = 32.
// 16×16 workgroup (256 threads). Each thread computes 1×4 outputs.
// Halves wasted M-dimension work vs 32×64 tile for small M.
// ═══════════════════════════════════════════════════════════════════

@group(0) @binding(0) var<storage, read> smat_a: array<f32>;
@group(0) @binding(1) var<storage, read> smat_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> smat_c: array<f32>;
@group(0) @binding(3) var<uniform> smat_params: MatmulParams;

const STILE_M: u32 = 16;
const STILE_N: u32 = 64;
const STILE_K: u32 = 32;
const STILE_A_STRIDE: u32 = 33;
const STILE_B_STRIDE: u32 = 65;
var<workgroup> stile_a: array<f32, 528>;   // [16, 33]
var<workgroup> stile_b: array<f32, 2080>;  // [32, 65]

@compute @workgroup_size(16, 16)
fn matmul_small(@builtin(global_invocation_id) gid: vec3<u32>,
                @builtin(local_invocation_id) lid: vec3<u32>) {
    let lr = lid.x;
    let lc = lid.y;
    let M = smat_params.M;
    let N = smat_params.N;
    let K = smat_params.K;

    let wg_row = gid.x - lid.x;
    let wg_col = gid.y * 4u - lid.y * 4u;
    let row0 = wg_row + lr;
    let col0 = wg_col + lc * 4u;

    var acc0: f32 = 0.0; var acc1: f32 = 0.0; var acc2: f32 = 0.0; var acc3: f32 = 0.0;
    let num_tiles = (K + STILE_K - 1u) / STILE_K;
    let lin = lr * 16u + lc;

    for (var t: u32 = 0u; t < num_tiles; t++) {
        let tk = t * STILE_K;

        // Load tile_a[16, 32]: 256 threads, 2 elements each
        for (var i: u32 = 0u; i < 2u; i++) {
            let idx = lin * 2u + i;
            let tr = idx / STILE_K;
            let tc = idx % STILE_K;
            let a_row = wg_row + tr;
            let a_col = tk + tc;
            if (a_row < M && a_col < K) {
                stile_a[tr * STILE_A_STRIDE + tc] = smat_a[a_row * K + a_col];
            } else {
                stile_a[tr * STILE_A_STRIDE + tc] = 0.0;
            }
        }

        // Load tile_b[32, 64]: 256 threads, 8 elements each
        for (var i: u32 = 0u; i < 8u; i++) {
            let idx = lin * 8u + i;
            let tr = idx / STILE_N;
            let tc = idx % STILE_N;
            let b_row = tk + tr;
            let b_col = wg_col + tc;
            if (b_row < K && b_col < N) {
                stile_b[tr * STILE_B_STRIDE + tc] = smat_b[b_row * N + b_col];
            } else {
                stile_b[tr * STILE_B_STRIDE + tc] = 0.0;
            }
        }

        workgroupBarrier();

        let c0 = lc * 4u;
        for (var k: u32 = 0u; k < STILE_K; k++) {
            let a_val = stile_a[lr * STILE_A_STRIDE + k];
            acc0 += a_val * stile_b[k * STILE_B_STRIDE + c0];
            acc1 += a_val * stile_b[k * STILE_B_STRIDE + c0 + 1u];
            acc2 += a_val * stile_b[k * STILE_B_STRIDE + c0 + 2u];
            acc3 += a_val * stile_b[k * STILE_B_STRIDE + c0 + 3u];
        }
        workgroupBarrier();
    }

    if (row0 < M && col0 < N) { smat_c[row0 * N + col0] = acc0; }
    if (row0 < M && col0 + 1u < N) { smat_c[row0 * N + col0 + 1u] = acc1; }
    if (row0 < M && col0 + 2u < N) { smat_c[row0 * N + col0 + 2u] = acc2; }
    if (row0 < M && col0 + 3u < N) { smat_c[row0 * N + col0 + 3u] = acc3; }
}
