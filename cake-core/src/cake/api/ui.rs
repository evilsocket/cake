use std::collections::HashSet;
use std::sync::Arc;

use actix_web::{web, HttpRequest, HttpResponse};
use serde::Serialize;
use tokio::sync::RwLock;

use crate::models::{ImageGenerator, TextGenerator};

use super::Master;

const INDEX_HTML: &str = include_str!("index.html");

/// Serve the single-page UI.
pub async fn index<TG, IG>(
    state: web::Data<Arc<RwLock<Master<TG, IG>>>>,
    req: HttpRequest,
) -> HttpResponse
where
    TG: TextGenerator + Send + Sync + 'static,
    IG: ImageGenerator + Send + Sync + 'static,
{
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
struct TopologyResponse {
    model: String,
    dtype: String,
    num_layers: usize,
    memory_bytes: u64,
    layer_size_bytes: u64,
    master: MasterInfo,
    workers: Vec<WorkerTopologyInfo>,
}

#[derive(Serialize)]
struct MasterInfo {
    device: String,
    layers: Vec<String>,
}

#[derive(Serialize)]
struct WorkerTopologyInfo {
    name: String,
    host: String,
    description: Option<String>,
    layers: Vec<String>,
}

/// Return cluster topology as JSON.
pub async fn topology<TG, IG>(
    state: web::Data<Arc<RwLock<Master<TG, IG>>>>,
    req: HttpRequest,
) -> HttpResponse
where
    TG: TextGenerator + Send + Sync + 'static,
    IG: ImageGenerator + Send + Sync + 'static,
{
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
        });
    }

    // Master layers = all layers not assigned to workers
    let master_layers: Vec<String> = (0..num_layers)
        .map(|i| format!("model.layers.{i}"))
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

    let response = TopologyResponse {
        model: TG::MODEL_NAME.to_string(),
        dtype: format!("{:?}", ctx.dtype),
        num_layers,
        memory_bytes,
        layer_size_bytes,
        master: MasterInfo {
            device: device.to_string(),
            layers: master_layers,
        },
        workers,
    };

    HttpResponse::Ok().json(response)
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
                let shards: HashSet<&str> =
                    weight_map.values().filter_map(|v| v.as_str()).collect();
                let total: u64 = shards
                    .iter()
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
async fn check_ui_auth<TG, IG>(
    state: &web::Data<Arc<RwLock<Master<TG, IG>>>>,
    req: &HttpRequest,
) -> bool
where
    TG: TextGenerator + Send + Sync + 'static,
    IG: ImageGenerator + Send + Sync + 'static,
{
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
