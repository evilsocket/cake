// ─── Elementwise compute shaders for Vulkan backend ─────────────────
// Each kernel operates on contiguous f32 buffers.
// Dispatched with workgroup_size(256) — one thread per element.

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
    let sigmoid_g = 1.0 / (1.0 + exp(-g));
    output[idx] = g * sigmoid_g * input_b[idx];
}

// ─── add: a + b ─────────────────────────────────────────────────────
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

// ─── sub_mul: (a - b) * c ──────────────────────────────────────────
// Uses input_a=a, input_b=b, and a third buffer for c.
@group(0) @binding(4) var<storage, read> input_c: array<f32>;

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

// ─── stable_softplus: max(x, ln(1 + exp(clamp(x, -inf, 88)))) ─────
@compute @workgroup_size(256)
fn stable_softplus(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) { return; }
    let x = input_a[idx];
    let clamped = min(x, 88.0);
    let sp = log(exp(clamped) + 1.0);
    output[idx] = max(x, sp);
}
