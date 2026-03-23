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
// Input vector x is loaded in tiles into shared memory so all 4
// column groups share the same cached data.
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
// First 256 floats: shared x tile; next 256: reduction scratch
var<workgroup> gemv_shared: array<f32, 512>;

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

    // Process K in tiles of 256 (one element per thread loaded into shared x cache)
    var partial: f32 = 0.0;
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

        // Each thread in its column group processes a stride of the tile
        if (col < N) {
            let tile_base = t * 256u;
            let tile_end = min(256u, K - tile_base);
            var k = lane;
            while (k < tile_end) {
                partial += gemv_shared[k] * gemv_w[(tile_base + k) * N + col];
                k += GEMV_GROUP;
            }
        }
        workgroupBarrier();
    }

    // Store partial sum in reduction scratch area (offset 256)
    let red_base = 256u + col_idx * GEMV_GROUP;
    gemv_shared[red_base + lane] = partial;
    workgroupBarrier();

    // Parallel reduction within each 64-element group
    if (lane < 32u) { gemv_shared[red_base + lane] += gemv_shared[red_base + lane + 32u]; }
    workgroupBarrier();
    if (lane < 16u) { gemv_shared[red_base + lane] += gemv_shared[red_base + lane + 16u]; }
    workgroupBarrier();
    if (lane < 8u) { gemv_shared[red_base + lane] += gemv_shared[red_base + lane + 8u]; }
    workgroupBarrier();
    if (lane < 4u) { gemv_shared[red_base + lane] += gemv_shared[red_base + lane + 4u]; }
    workgroupBarrier();
    if (lane < 2u) { gemv_shared[red_base + lane] += gemv_shared[red_base + lane + 2u]; }
    workgroupBarrier();
    if (lane == 0u && col < N) {
        gemv_y[col] = gemv_shared[red_base] + gemv_shared[red_base + 1u];
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tiled GEMM: C[M,N] = A[M,K] × B[K,N]  (for M > 1)
// 16×16 output tile with 2×2 register tiling, K-tile = 32.
// 8×8 workgroup (64 threads). Doubled K-tile halves the number of
// tile iterations and barrier synchronizations.
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

const TILE_MN: u32 = 16;   // output tile M and N dimension
const TILE_K: u32 = 32;    // K dimension tile (doubled for fewer iterations)
const TILE_A_STRIDE: u32 = 33;  // padded stride to avoid bank conflicts (32 banks on RDNA 2)
var<workgroup> tile_a: array<f32, 528>;  // [16, 33] padded
var<workgroup> tile_b: array<f32, 512>;  // [32, 16]

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

    // Workgroup base position
    let wg_row = gid.x * 2u - lid.x * 2u;
    let wg_col = gid.y * 2u - lid.y * 2u;

    // 2×2 accumulators
    var acc00: f32 = 0.0;
    var acc01: f32 = 0.0;
    var acc10: f32 = 0.0;
    var acc11: f32 = 0.0;
    let num_tiles = (K + TILE_K - 1u) / TILE_K;

    // Linear thread index (64 threads load 512 elements = 8 per thread)
    let lin = lr * 8u + lc;

    for (var t: u32 = 0u; t < num_tiles; t++) {
        let tk = t * TILE_K;

        // Cooperative load tile_a[16, 32] with padded stride: 64 threads, 8 elements each
        for (var i: u32 = 0u; i < 8u; i++) {
            let idx = lin * 8u + i;
            let tr = idx / TILE_K;   // row 0..15
            let tc = idx % TILE_K;   // col 0..31
            let a_row = wg_row + tr;
            let a_col = tk + tc;
            if (a_row < M && a_col < K) {
                tile_a[tr * TILE_A_STRIDE + tc] = mat_a[a_row * K + a_col];
            } else {
                tile_a[tr * TILE_A_STRIDE + tc] = 0.0;
            }
        }

        // Cooperative load tile_b[32, 16]: 64 threads, 8 elements each
        for (var i: u32 = 0u; i < 8u; i++) {
            let idx = lin * 8u + i;
            let tr = idx / TILE_MN;  // row 0..31
            let tc = idx % TILE_MN;  // col 0..15
            let b_row = tk + tr;
            let b_col = wg_col + tc;
            if (b_row < K && b_col < N) {
                tile_b[idx] = mat_b[b_row * N + b_col];
            } else {
                tile_b[idx] = 0.0;
            }
        }

        workgroupBarrier();

        // Each thread accumulates its 2×2 block over K-tile of 32
        let r0 = lr * 2u;
        let r1 = r0 + 1u;
        let c0 = lc * 2u;
        let c1 = c0 + 1u;
        for (var k: u32 = 0u; k < TILE_K; k++) {
            let a0k = tile_a[r0 * TILE_A_STRIDE + k];
            let a1k = tile_a[r1 * TILE_A_STRIDE + k];
            let bk0 = tile_b[k * TILE_MN + c0];
            let bk1 = tile_b[k * TILE_MN + c1];
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
