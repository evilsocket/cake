use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use actix_web::{web, HttpRequest, HttpResponse};
use rayon::prelude::*;
use safetensors::SafeTensors;
use serde::Serialize;
use tokio::sync::RwLock;

use crate::cake::discovery;
use crate::models::Model;

use crate::cake::Master;

const INDEX_HTML: &str = include_str!("index.html");

/// Serve the single-page UI.
pub async fn index<M: Model>(
    state: web::Data<Arc<RwLock<Master<M>>>>,
    req: HttpRequest,
) -> HttpResponse {
    if !check_ui_auth(&state, &req).await {
        return HttpResponse::Unauthorized()
            .insert_header(("WWW-Authenticate", "Basic realm=\"cake\""))
            .body("Unauthorized");
    }
    HttpResponse::Ok()
        .content_type("text/html; charset=utf-8")
        .body(INDEX_HTML)
}

#[derive(Serialize)]
struct TensorDetail {
    name: String,
    short_name: String,
    dtype: String,
    shape: Vec<usize>,
    size_bytes: u64,
}

#[derive(Serialize)]
struct LayerDetail {
    total_bytes: u64,
    tensors: Vec<TensorDetail>,
}

#[derive(Serialize)]
struct TopologyResponse {
    model: String,
    model_id: String,
    dtype: String,
    num_layers: usize,
    memory_bytes: u64,
    layer_size_bytes: u64,
    master: MasterInfo,
    workers: Vec<WorkerTopologyInfo>,
    layer_details: HashMap<String, LayerDetail>,
}

#[derive(Serialize)]
struct MasterInfo {
    device: String,
    backend: String,
    layers: Vec<String>,
    vram_bytes: u64,
    tflops: f64,
    hostname: String,
    os: String,
}

#[derive(Serialize)]
struct WorkerTopologyInfo {
    name: String,
    host: String,
    description: Option<String>,
    layers: Vec<String>,
    vram_bytes: u64,
    tflops: f64,
    backend: String,
    hostname: String,
    os: String,
}

/// Return cluster topology as JSON.
pub async fn topology<M: Model>(
    state: web::Data<Arc<RwLock<Master<M>>>>,
    req: HttpRequest,
) -> HttpResponse {
    if !check_ui_auth(&state, &req).await {
        return HttpResponse::Unauthorized()
            .insert_header(("WWW-Authenticate", "Basic realm=\"cake\""))
            .body("Unauthorized");
    }

    let master = state.read().await;
    let ctx = &master.ctx;

    let num_layers = ctx
        .config
        .as_ref()
        .map(|c| c.num_hidden_layers)
        .unwrap_or(0);

    // Estimate per-layer size from safetensors index
    let layer_size_bytes = estimate_layer_size(&ctx.data_path, num_layers);

    // Collect worker layers from topology
    let mut worker_layer_set = HashSet::new();
    let mut workers = Vec::new();

    for (name, node) in ctx.topology.iter() {
        for layer in &node.layers {
            worker_layer_set.insert(layer.clone());
        }
        workers.push(WorkerTopologyInfo {
            name: name.clone(),
            host: node.host.clone(),
            description: node.description.clone(),
            layers: node.layers.clone(),
            vram_bytes: node.vram_bytes,
            tflops: node.tflops,
            backend: node.backend.clone(),
            hostname: node.hostname.clone(),
            os: node.os.clone(),
        });
    }

    // Derive layer prefix from config
    let layer_prefix = ctx
        .config
        .as_ref()
        .map(|c| format!("{}.layers", c.model_prefix))
        .unwrap_or_else(|| "model.layers".to_string());

    // Master layers = all layers not assigned to workers
    let master_layers: Vec<String> = (0..num_layers)
        .map(|i| format!("{layer_prefix}.{i}"))
        .filter(|l| !worker_layer_set.contains(l))
        .collect();

    let device = if ctx.device.is_cuda() {
        "cuda"
    } else if ctx.device.is_metal() {
        "metal"
    } else {
        "cpu"
    };

    let memory_bytes = memory_stats::memory_stats()
        .map(|s| s.physical_mem as u64)
        .unwrap_or(0);

    let master_gpus = discovery::detect_gpus();
    let master_vram: u64 = master_gpus.iter().map(|g| g.vram_bytes).sum();
    let master_tflops: f64 = master_gpus.iter().map(|g| g.tflops as f64).sum();

    let backend = if ctx.device.is_cuda() {
        discovery::detect_cuda_version().unwrap_or_else(|| "CUDA".to_string())
    } else if ctx.device.is_metal() {
        // Use GPU name which now contains Apple chip model
        master_gpus.first().map(|g| g.name.clone()).unwrap_or_else(|| "Metal".to_string())
    } else {
        "CPU".to_string()
    };

    let layer_details = read_layer_tensor_details(&ctx.data_path, num_layers, &layer_prefix);

    let response = TopologyResponse {
        model: if master.model.is_some() {
            M::MODEL_NAME.to_string()
        } else {
            "none".to_string()
        },
        model_id: {
            let m = ctx.args.model.trim_end_matches('/');
            // If it looks like a HF repo ID (org/name, no dots or path seps at start), keep as-is
            let parts: Vec<&str> = m.split('/').collect();
            if parts.len() == 2 && !m.starts_with('.') && !m.starts_with('/') {
                m.to_string()
            } else {
                // Local path: extract just the last component
                parts.last().unwrap_or(&m).to_string()
            }
        },
        dtype: format!("{:?}", ctx.dtype),
        num_layers,
        memory_bytes,
        layer_size_bytes,
        master: MasterInfo {
            device: device.to_string(),
            backend,
            layers: master_layers,
            vram_bytes: master_vram,
            tflops: master_tflops,
            hostname: discovery::detect_hostname(),
            os: std::env::consts::OS.to_string(),
        },
        workers,
        layer_details,
    };

    HttpResponse::Ok().json(response)
}

/// Read per-layer tensor metadata from safetensors file headers (via mmap).
/// Only the header bytes are paged in — tensor data is never accessed.
fn read_layer_tensor_details(
    data_path: &std::path::Path,
    num_layers: usize,
    layer_prefix: &str,
) -> HashMap<String, LayerDetail> {
    // Collect safetensors shard files
    let index_path = data_path.join("model.safetensors.index.json");
    let shard_files: Vec<std::path::PathBuf> = if let Ok(data) =
        std::fs::read_to_string(&index_path)
    {
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&data) {
            if let Some(weight_map) = json.get("weight_map").and_then(|v| v.as_object()) {
                let shards: HashSet<&str> =
                    weight_map.values().filter_map(|v| v.as_str()).collect();
                shards.iter().map(|s| data_path.join(s)).collect()
            } else {
                vec![]
            }
        } else {
            vec![]
        }
    } else {
        let single = data_path.join("model.safetensors");
        if single.exists() {
            vec![single]
        } else {
            vec![]
        }
    };

    // Read headers from each shard via mmap (parallel)
    let all_tensors: Vec<(String, u64, String, Vec<usize>)> = shard_files
        .par_iter()
        .flat_map(|shard_path| {
            let mut entries = Vec::new();
            let file = match std::fs::File::open(shard_path) {
                Ok(f) => f,
                Err(_) => return entries,
            };
            let buffer = match unsafe { memmap2::MmapOptions::new().map(&file) } {
                Ok(b) => b,
                Err(_) => return entries,
            };
            let tensors = match SafeTensors::deserialize(&buffer) {
                Ok(t) => t,
                Err(_) => return entries,
            };

            for name in tensors.names() {
                if let Ok(tv) = tensors.tensor(name) {
                    let shape: Vec<usize> = tv.shape().to_vec();
                    let size_bytes = tv.data().len() as u64;
                    let dtype = format!("{:?}", tv.dtype());
                    entries.push((name.to_string(), size_bytes, dtype, shape));
                }
            }
            entries
        })
        .collect();

    // Group by layer (parallel)
    let result: HashMap<String, LayerDetail> = (0..num_layers)
        .into_par_iter()
        .filter_map(|layer_idx| {
            let layer_name = format!("{layer_prefix}.{layer_idx}");
            let dot_prefix = format!("{layer_name}.");

            let mut tensors: Vec<TensorDetail> = Vec::new();
            let mut total_bytes: u64 = 0;

            for (tensor_name, size_bytes, dtype, shape) in &all_tensors {
                if tensor_name.starts_with(&dot_prefix) {
                    let short_name = tensor_name[dot_prefix.len()..].to_string();
                    tensors.push(TensorDetail {
                        name: tensor_name.clone(),
                        short_name,
                        dtype: dtype.clone(),
                        shape: shape.clone(),
                        size_bytes: *size_bytes,
                    });
                    total_bytes += size_bytes;
                }
            }

            tensors.sort_by(|a, b| a.name.cmp(&b.name));

            if tensors.is_empty() {
                None
            } else {
                Some((layer_name, LayerDetail { total_bytes, tensors }))
            }
        })
        .collect();

    result
}

/// Estimate average layer size in bytes by summing safetensors shard file sizes
/// and dividing by number of layers.
fn estimate_layer_size(data_path: &std::path::Path, num_layers: usize) -> u64 {
    if num_layers == 0 {
        return 0;
    }

    // Try sharded model first
    let index_path = data_path.join("model.safetensors.index.json");
    if let Ok(data) = std::fs::read_to_string(&index_path) {
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&data) {
            if let Some(weight_map) = json.get("weight_map").and_then(|v| v.as_object()) {
                let shards: Vec<&str> =
                    weight_map.values().filter_map(|v| v.as_str()).collect::<HashSet<_>>().into_iter().collect();
                let total: u64 = shards
                    .par_iter()
                    .filter_map(|s| std::fs::metadata(data_path.join(s)).ok())
                    .map(|m| m.len())
                    .sum();
                return total / num_layers as u64;
            }
        }
    }

    // Single safetensors file
    let single = data_path.join("model.safetensors");
    if let Ok(m) = std::fs::metadata(&single) {
        return m.len() / num_layers as u64;
    }

    0
}

/// Check basic auth if `--ui-auth` is configured. Returns true if OK.
async fn check_ui_auth<M: Model>(
    state: &web::Data<Arc<RwLock<Master<M>>>>,
    req: &HttpRequest,
) -> bool {
    let master = state.read().await;
    let expected = match &master.ctx.args.ui_auth {
        Some(cred) => cred.clone(),
        None => return true, // no auth configured
    };

    let header = match req.headers().get("Authorization") {
        Some(h) => h.to_str().unwrap_or(""),
        None => return false,
    };

    if let Some(encoded) = header.strip_prefix("Basic ") {
        if let Ok(decoded) = base64::Engine::decode(
            &base64::engine::general_purpose::STANDARD,
            encoded.trim(),
        ) {
            if let Ok(cred_str) = String::from_utf8(decoded) {
                return cred_str == expected;
            }
        }
    }

    false
}
