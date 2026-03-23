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
// 16×16 tiles with 2×2 register tiling: 8×8 workgroup, each thread
// computes a 2×2 block of outputs for higher arithmetic intensity.
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

@compute @workgroup_size(8, 8)
fn matmul(@builtin(global_invocation_id) gid: vec3<u32>,
          @builtin(local_invocation_id) lid: vec3<u32>) {
    // Each thread covers a 2×2 block of the output tile
    let row0 = gid.x * 2u;
    let col0 = gid.y * 2u;
    let lr = lid.x;
    let lc = lid.y;
    let M = mat_params.M;
    let N = mat_params.N;
    let K = mat_params.K;

    // 2×2 accumulators
    var acc00: f32 = 0.0;
    var acc01: f32 = 0.0;
    var acc10: f32 = 0.0;
    var acc11: f32 = 0.0;
    let num_tiles = (K + TILE - 1) / TILE;

    // Linear thread index for cooperative tile loading (64 threads load 256 elements = 4 each)
    let lin = lr * 8u + lc;

    for (var t: u32 = 0; t < num_tiles; t++) {
        let tile_k = t * TILE;

        // Cooperative load: 64 threads load 16×16 = 256 elements, 4 per thread
        for (var i: u32 = 0u; i < 4u; i++) {
            let idx = lin * 4u + i;
            let tr = idx / TILE;
            let tc = idx % TILE;

            // Load tile_a: row = workgroup_row_base + tr, col = tile_k + tc
            let a_row = gid.x * 2u - lid.x * 2u + tr;  // workgroup base row + tr
            let a_col = tile_k + tc;
            if (a_row < M && a_col < K) {
                tile_a[tr * TILE + tc] = mat_a[a_row * K + a_col];
            } else {
                tile_a[tr * TILE + tc] = 0.0;
            }

            // Load tile_b: row = tile_k + tr, col = workgroup_col_base + tc
            let b_row = tile_k + tr;
            let b_col = gid.y * 2u - lid.y * 2u + tc;  // workgroup base col + tc
            if (b_row < K && b_col < N) {
                tile_b[tr * TILE + tc] = mat_b[b_row * N + b_col];
            } else {
                tile_b[tr * TILE + tc] = 0.0;
            }
        }

        workgroupBarrier();

        // Each thread accumulates its 2×2 block
        let r0 = lr * 2u;
        let r1 = r0 + 1u;
        let c0 = lc * 2u;
        let c1 = c0 + 1u;
        for (var k: u32 = 0; k < TILE; k++) {
            let a0k = tile_a[r0 * TILE + k];
            let a1k = tile_a[r1 * TILE + k];
            let bk0 = tile_b[k * TILE + c0];
            let bk1 = tile_b[k * TILE + c1];
            acc00 += a0k * bk0;
            acc01 += a0k * bk1;
            acc10 += a1k * bk0;
            acc11 += a1k * bk1;
        }
        workgroupBarrier();
    }

    // Write 2×2 output block
    if (row0 < M && col0 < N) { mat_c[row0 * N + col0] = acc00; }
    if (row0 < M && col0 + 1u < N) { mat_c[row0 * N + col0 + 1u] = acc01; }
    if (row0 + 1u < M && col0 < N) { mat_c[(row0 + 1u) * N + col0] = acc10; }
    if (row0 + 1u < M && col0 + 1u < N) { mat_c[(row0 + 1u) * N + col0 + 1u] = acc11; }
}
