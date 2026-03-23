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
// 32×32 output tile with 4×4 register blocking: 8×8 workgroup (64
// threads = 1 RDNA 2 wavefront), each thread computes a 4×4 block.
// K-dimension tile is 16. Shared memory: 32×16 + 16×32 = 1024 floats.
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

const TILE_MN: u32 = 32;  // output tile M and N dimension
const TILE_K: u32 = 16;   // K dimension tile
var<workgroup> tile_a: array<f32, 512>;  // [32, 16]
var<workgroup> tile_b: array<f32, 512>;  // [16, 32]

@compute @workgroup_size(8, 8)
fn matmul(@builtin(global_invocation_id) gid: vec3<u32>,
          @builtin(local_invocation_id) lid: vec3<u32>,
          @builtin(workgroup_id) wid: vec3<u32>) {
    let lr = lid.x;
    let lc = lid.y;
    let M = mat_params.M;
    let N = mat_params.N;
    let K = mat_params.K;

    // Workgroup base row/col in output
    let wg_row = wid.x * TILE_MN;
    let wg_col = wid.y * TILE_MN;

    // This thread's 4×4 output block position
    let row0 = wg_row + lr * 4u;
    let col0 = wg_col + lc * 4u;

    // 4×4 accumulators
    var acc: array<f32, 16>;
    for (var i = 0u; i < 16u; i++) { acc[i] = 0.0; }

    let num_tiles = (K + TILE_K - 1u) / TILE_K;
    let lin = lr * 8u + lc;  // linear thread index 0..63

    for (var t: u32 = 0u; t < num_tiles; t++) {
        let tile_k = t * TILE_K;

        // Cooperative load: 64 threads load 512 elements (8 per thread) for each tile
        for (var i: u32 = 0u; i < 8u; i++) {
            let idx = lin * 8u + i;
            let tr = idx / TILE_K;  // row within 32×16 tile
            let tc = idx % TILE_K;  // col within 32×16 tile

            let a_row = wg_row + tr;
            let a_col = tile_k + tc;
            if (a_row < M && a_col < K) {
                tile_a[idx] = mat_a[a_row * K + a_col];
            } else {
                tile_a[idx] = 0.0;
            }
        }
        for (var i: u32 = 0u; i < 8u; i++) {
            let idx = lin * 8u + i;
            let tr = idx / TILE_MN;  // row within 16×32 tile
            let tc = idx % TILE_MN;  // col within 16×32 tile

            let b_row = tile_k + tr;
            let b_col = wg_col + tc;
            if (b_row < K && b_col < N) {
                tile_b[idx] = mat_b[b_row * N + b_col];
            } else {
                tile_b[idx] = 0.0;
            }
        }

        workgroupBarrier();

        // Each thread accumulates its 4×4 block
        let r_base = lr * 4u;
        let c_base = lc * 4u;
        for (var k: u32 = 0u; k < TILE_K; k++) {
            // Load 4 A values from this thread's rows
            let a0 = tile_a[(r_base) * TILE_K + k];
            let a1 = tile_a[(r_base + 1u) * TILE_K + k];
            let a2 = tile_a[(r_base + 2u) * TILE_K + k];
            let a3 = tile_a[(r_base + 3u) * TILE_K + k];
            // Load 4 B values from this thread's cols
            let b0 = tile_b[k * TILE_MN + c_base];
            let b1 = tile_b[k * TILE_MN + c_base + 1u];
            let b2 = tile_b[k * TILE_MN + c_base + 2u];
            let b3 = tile_b[k * TILE_MN + c_base + 3u];
            // 4×4 outer product
            acc[0]  += a0 * b0; acc[1]  += a0 * b1; acc[2]  += a0 * b2; acc[3]  += a0 * b3;
            acc[4]  += a1 * b0; acc[5]  += a1 * b1; acc[6]  += a1 * b2; acc[7]  += a1 * b3;
            acc[8]  += a2 * b0; acc[9]  += a2 * b1; acc[10] += a2 * b2; acc[11] += a2 * b3;
            acc[12] += a3 * b0; acc[13] += a3 * b1; acc[14] += a3 * b2; acc[15] += a3 * b3;
        }
        workgroupBarrier();
    }

    // Write 4×4 output block
    for (var dr: u32 = 0u; dr < 4u; dr++) {
        for (var dc: u32 = 0u; dc < 4u; dc++) {
            let r = row0 + dr;
            let c = col0 + dc;
            if (r < M && c < N) {
                mat_c[r * N + c] = acc[dr * 4u + dc];
            }
        }
    }
}
