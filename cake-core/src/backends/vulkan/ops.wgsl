// ─── WGSL compute shaders for Vulkan backend ───────────────────────
// All kernels operate on contiguous f32 storage buffers.

// ═══════════════════════════════════════════════════════════════════
// Shared bindings for elementwise ops (2-input)
// ═══════════════════════════════════════════════════════════════════

@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

struct Params {
    count: u32,
}

// ─── silu_mul: silu(a) * b ──────────────────────────────────────────
@compute @workgroup_size(256)
fn silu_mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) { return; }
    let g = input_a[idx];
    output[idx] = g / (1.0 + exp(-g)) * input_b[idx];
}

// ─── add2: a + b ────────────────────────────────────────────────────
@compute @workgroup_size(256)
fn add2(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) { return; }
    output[idx] = input_a[idx] + input_b[idx];
}

// ─── exp_mul: a * exp(b) ────────────────────────────────────────────
@compute @workgroup_size(256)
fn exp_mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) { return; }
    output[idx] = input_a[idx] * exp(input_b[idx]);
}

// ─── stable_softplus: max(x, ln(1 + exp(min(x, 88)))) ─────────────
@compute @workgroup_size(256)
fn stable_softplus(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) { return; }
    let x = input_a[idx];
    let sp = log(exp(min(x, 88.0)) + 1.0);
    output[idx] = max(x, sp);
}

// ═══════════════════════════════════════════════════════════════════
// 3-input ops (extra binding for input_c)
// ═══════════════════════════════════════════════════════════════════

@group(0) @binding(4) var<storage, read> input_c: array<f32>;

// ─── sub_mul: (a - b) * c ──────────────────────────────────────────
@compute @workgroup_size(256)
fn sub_mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) { return; }
    output[idx] = (input_a[idx] - input_b[idx]) * input_c[idx];
}

// ─── add3: a + b + c ───────────────────────────────────────────────
@compute @workgroup_size(256)
fn add3(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) { return; }
    output[idx] = input_a[idx] + input_b[idx] + input_c[idx];
}

// ═══════════════════════════════════════════════════════════════════
// Tiled matrix multiplication: C[M,N] = A[M,K] × B[K,N]
//
// Uses 16×16 tiles with workgroup-shared memory. Each workgroup
// computes one 16×16 output tile by iterating over K in chunks of 16.
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

var<workgroup> tile_a: array<f32, 256>; // 16×16
var<workgroup> tile_b: array<f32, 256>; // 16×16

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
        // Load tile of A: rows [row], cols [t*TILE + lc]
        let a_col = t * TILE + lc;
        if (row < M && a_col < K) {
            tile_a[lr * TILE + lc] = mat_a[row * K + a_col];
        } else {
            tile_a[lr * TILE + lc] = 0.0;
        }

        // Load tile of B: rows [t*TILE + lr], cols [col]
        let b_row = t * TILE + lr;
        if (b_row < K && col < N) {
            tile_b[lr * TILE + lc] = mat_b[b_row * N + col];
        } else {
            tile_b[lr * TILE + lc] = 0.0;
        }

        workgroupBarrier();

        // Accumulate dot product for this tile
        for (var k: u32 = 0; k < TILE; k++) {
            acc += tile_a[lr * TILE + k] * tile_b[k * TILE + lc];
        }

        workgroupBarrier();
    }

    if (row < M && col < N) {
        mat_c[row * N + col] = acc;
    }
}
