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
const TILE_K: u32 = 64;   // K dimension tile (doubled for fewer iterations)
const TILE_A_STRIDE: u32 = 65;  // padded stride (64+1)
const TILE_B_STRIDE: u32 = 65;  // padded stride (64+1)
var<workgroup> tile_a: array<f32, 2080>;  // [32, 65] padded
var<workgroup> tile_b: array<f32, 4160>;  // [64, 65] padded

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

        // Cooperative load tile_a[32, 64]: 256 threads, 8 elements each
        for (var i: u32 = 0u; i < 8u; i++) {
            let idx = lin * 8u + i;
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

        // Cooperative load tile_b[64, 64]: 256 threads, 16 elements each
        for (var i: u32 = 0u; i < 16u; i++) {
            let idx = lin * 16u + i;
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
