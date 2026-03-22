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
// Each workgroup computes one output element y[j].
// 256 threads cooperatively reduce the K-length dot product.
// Uses shared memory for the parallel reduction.
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
var<workgroup> gemv_shared: array<f32, 256>;

@compute @workgroup_size(256)
fn gemv(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let col = wid.x;           // which output element
    let tid = lid.x;           // thread within workgroup
    let N = gemv_params.N;
    let K = gemv_params.K;

    if (col >= N) { return; }

    // Each thread accumulates a partial sum over a strided chunk of K
    var partial: f32 = 0.0;
    var k = tid;
    while (k < K) {
        partial += gemv_x[k] * gemv_w[k * N + col];
        k += GEMV_WG;
    }
    gemv_shared[tid] = partial;
    workgroupBarrier();

    // Parallel reduction in shared memory
    if (tid < 128u) { gemv_shared[tid] += gemv_shared[tid + 128u]; }
    workgroupBarrier();
    if (tid < 64u) { gemv_shared[tid] += gemv_shared[tid + 64u]; }
    workgroupBarrier();
    if (tid < 32u) { gemv_shared[tid] += gemv_shared[tid + 32u]; }
    workgroupBarrier();
    if (tid < 16u) { gemv_shared[tid] += gemv_shared[tid + 16u]; }
    workgroupBarrier();
    if (tid < 8u) { gemv_shared[tid] += gemv_shared[tid + 8u]; }
    workgroupBarrier();
    if (tid < 4u) { gemv_shared[tid] += gemv_shared[tid + 4u]; }
    workgroupBarrier();
    if (tid < 2u) { gemv_shared[tid] += gemv_shared[tid + 2u]; }
    workgroupBarrier();
    if (tid == 0u) {
        gemv_y[col] = gemv_shared[0] + gemv_shared[1];
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
