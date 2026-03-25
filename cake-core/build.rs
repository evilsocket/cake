fn main() {
    #[cfg(feature = "metal")]
    {
        println!("cargo:rustc-link-lib=framework=CoreGraphics");
    }
    #[cfg(feature = "vulkan")]
    {
        println!("cargo::rerun-if-changed=src/backends/vulkan/ops.wgsl");

        let wgsl_src = std::fs::read_to_string("src/backends/vulkan/ops.wgsl")
            .expect("failed to read ops.wgsl");
        let module = naga::front::wgsl::parse_str(&wgsl_src)
            .expect("failed to parse WGSL");
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .expect("WGSL validation failed");

        let options = naga::back::spv::Options {
            lang_version: (1, 3),
            ..Default::default()
        };
        // Generate one SPIR-V module per entry point
        let entry_points: Vec<String> = module
            .entry_points
            .iter()
            .map(|ep| ep.name.clone())
            .collect();

        let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
        let mut includes = String::new();

        for ep_name in &entry_points {
            let pipeline_options = naga::back::spv::PipelineOptions {
                shader_stage: naga::ShaderStage::Compute,
                entry_point: ep_name.clone(),
            };
            let spv_words = naga::back::spv::write_vec(
                &module,
                &info,
                &options,
                Some(&pipeline_options),
            )
            .unwrap_or_else(|e| panic!("SPIR-V generation failed for {ep_name}: {e}"));

            // Write SPIR-V binary
            let spv_path = out_dir.join(format!("ops_{ep_name}.spv"));
            let bytes: Vec<u8> = spv_words.iter().flat_map(|w| w.to_le_bytes()).collect();
            std::fs::write(&spv_path, &bytes).unwrap();

            includes.push_str(&format!(
                "(\"{ep_name}\", include_bytes!(concat!(env!(\"OUT_DIR\"), \"/ops_{ep_name}.spv\"))),\n"
            ));
        }

        // Write a Rust file with all SPIR-V modules
        let rs_path = out_dir.join("spirv_ops.rs");
        let code = format!(
            "static SPIRV_MODULES: &[(&str, &[u8])] = &[\n{includes}];\n"
        );
        std::fs::write(&rs_path, &code).expect("failed to write SPIR-V module list");
    }

    #[cfg(feature = "cuda")]
    {
        println!("cargo::rerun-if-changed=src/backends/cuda/ops.cu");

        let builder = bindgen_cuda::Builder::default()
            .kernel_paths(vec!["src/backends/cuda/ops.cu"])
            .arg("--expt-relaxed-constexpr")
            .arg("-std=c++17")
            .arg("-O2");
        let bindings = builder.build_ptx().unwrap();

        let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
        bindings.write(out_dir.join("fused_ops_ptx.rs")).unwrap();
    }
}
