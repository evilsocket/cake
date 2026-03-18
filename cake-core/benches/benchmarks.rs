fn main() {
    divan::main();
}

mod bench_helpers;
mod bench_attention;
mod bench_mlp;
mod bench_linear_attn;
mod bench_moe;
mod bench_blocks;
mod bench_cache;
mod bench_serialization;
mod bench_quantization;
mod bench_auth;
mod bench_protocol;
mod bench_discovery;
mod bench_topology;
mod bench_utils;
mod bench_wav;
mod bench_flux;
mod bench_vibevoice;
#[cfg(feature = "luxtts")]
mod bench_luxtts;
