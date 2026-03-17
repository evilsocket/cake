fn main() {
    #[cfg(feature = "cuda")]
    {
        println!("cargo::rerun-if-changed=src/cuda/fused_ops.cu");

        let builder = bindgen_cuda::Builder::default()
            .kernel_paths(vec!["src/cuda/fused_ops.cu"])
            .arg("--expt-relaxed-constexpr")
            .arg("-std=c++17")
            .arg("-O2");
        let bindings = builder.build_ptx().unwrap();

        let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
        bindings.write(out_dir.join("fused_ops_ptx.rs")).unwrap();
    }
}
