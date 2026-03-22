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
    let i = gid.x;
    if (i >= params.count) { return; }
    let g = input_a[i];
    output[i] = g / (1.0 + exp(-g)) * input_b[i];
}

@compute @workgroup_size(256)
fn exp_mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) { return; }
    output[i] = input_a[i] * exp(input_b[i]);
}

@compute @workgroup_size(256)
fn stable_softplus(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) { return; }
    let x = input_a[i];
    output[i] = max(x, log(exp(min(x, 88.0)) + 1.0));
}

// ═══════════════════════════════════════════════════════════════════
// 3-input ops: extra binding 4=C
// ═══════════════════════════════════════════════════════════════════

@group(0) @binding(4) var<storage, read> input_c: array<f32>;

@compute @workgroup_size(256)
fn sub_mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) { return; }
    output[i] = (input_a[i] - input_b[i]) * input_c[i];
}

@compute @workgroup_size(256)
fn add3(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) { return; }
    output[i] = input_a[i] + input_b[i] + input_c[i];
}

// ═══════════════════════════════════════════════════════════════════
// GEMV: y[j] = sum_i(x[i] * W[i * N + j])  for x[1,K] × W[K,N] = y[1,N]
//
// Each workgroup computes 4 output elements y[wg*4 .. wg*4+3].
// 256 threads split into 4 groups of 64 (matching RDNA 2 wavefront).
// Each group reduces the K-length dot product for one output column.
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

const GEMV_WG: u32 = 256;
const GEMV_COLS: u32 = 4;       // output columns per workgroup
const GEMV_GROUP: u32 = 64;     // threads per column (= RDNA 2 wavefront)
var<workgroup> gemv_shared: array<f32, 256>;

@compute @workgroup_size(256)
fn gemv(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let tid = lid.x;
    let N = gemv_params.N;
    let K = gemv_params.K;

    // Map tid to (column_index, lane_within_group)
    let col_idx = tid / GEMV_GROUP;    // 0..3
    let lane = tid % GEMV_GROUP;       // 0..63
    let col = wid.x * GEMV_COLS + col_idx;

    // Accumulate partial dot product
    var partial: f32 = 0.0;
    if (col < N) {
        var k = lane;
        while (k < K) {
            partial += gemv_x[k] * gemv_w[k * N + col];
            k += GEMV_GROUP;
        }
    }

    // Store partial sum in shared memory
    let shared_base = col_idx * GEMV_GROUP;
    gemv_shared[shared_base + lane] = partial;
    workgroupBarrier();

    // Parallel reduction within each 64-element group
    if (lane < 32u) { gemv_shared[shared_base + lane] += gemv_shared[shared_base + lane + 32u]; }
    workgroupBarrier();
    if (lane < 16u) { gemv_shared[shared_base + lane] += gemv_shared[shared_base + lane + 16u]; }
    workgroupBarrier();
    if (lane < 8u) { gemv_shared[shared_base + lane] += gemv_shared[shared_base + lane + 8u]; }
    workgroupBarrier();
    if (lane < 4u) { gemv_shared[shared_base + lane] += gemv_shared[shared_base + lane + 4u]; }
    workgroupBarrier();
    if (lane < 2u) { gemv_shared[shared_base + lane] += gemv_shared[shared_base + lane + 2u]; }
    workgroupBarrier();
    if (lane == 0u && col < N) {
        gemv_y[col] = gemv_shared[shared_base] + gemv_shared[shared_base + 1u];
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tiled GEMM: C[M,N] = A[M,K] × B[K,N]  (for M > 1)
// 16×16 tiles with shared memory.
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

const TILE: u32 = 16;
var<workgroup> tile_a: array<f32, 256>;
var<workgroup> tile_b: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn matmul(@builtin(global_invocation_id) gid: vec3<u32>,
          @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = gid.x;
    let col = gid.y;
    let lr = lid.x;
    let lc = lid.y;
    let M = mat_params.M;
    let N = mat_params.N;
    let K = mat_params.K;

    var acc: f32 = 0.0;
    let num_tiles = (K + TILE - 1) / TILE;

    for (var t: u32 = 0; t < num_tiles; t++) {
        let a_col = t * TILE + lc;
        if (row < M && a_col < K) {
            tile_a[lr * TILE + lc] = mat_a[row * K + a_col];
        } else {
            tile_a[lr * TILE + lc] = 0.0;
        }

        let b_row = t * TILE + lr;
        if (b_row < K && col < N) {
            tile_b[lr * TILE + lc] = mat_b[b_row * N + col];
        } else {
            tile_b[lr * TILE + lc] = 0.0;
        }

        workgroupBarrier();
        for (var k: u32 = 0; k < TILE; k++) {
            acc += tile_a[lr * TILE + k] * tile_b[k * TILE + lc];
        }
        workgroupBarrier();
    }

    if (row < M && col < N) {
        mat_c[row * N + col] = acc;
    }
}
