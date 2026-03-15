mod image;
pub mod text;
mod ui;
pub mod video;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use actix_web::web;
use actix_web::App;
use actix_web::HttpResponse;
use actix_web::HttpServer;
use serde::Serialize;
use tokio::sync::{RwLock, Semaphore};

use crate::models::{ImageGenerator, TextGenerator, VideoGenerator};

use image::*;
use text::*;

use super::master::VideoMaster;
use super::Master;

/// Bounded request queue for backpressure.
/// Limits concurrent waiting requests and tracks queue depth.
pub struct RequestQueue {
    /// Semaphore limiting how many requests can wait concurrently.
    semaphore: Arc<Semaphore>,
    /// Current number of requests in the queue (waiting + processing).
    pending: Arc<AtomicUsize>,
    /// Maximum allowed pending requests.
    max_pending: usize,
}

impl RequestQueue {
    pub fn new(max_pending: usize) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(max_pending)),
            pending: Arc::new(AtomicUsize::new(0)),
            max_pending,
        }
    }

    /// Try to acquire a slot. Returns None if queue is full.
    /// The returned guard is `'static` and can be moved into spawned tasks.
    pub fn try_acquire(&self) -> Option<QueueGuard> {
        let permit = self.semaphore.clone().try_acquire_owned().ok()?;
        self.pending.fetch_add(1, Ordering::Relaxed);
        Some(QueueGuard {
            _permit: permit,
            pending: self.pending.clone(),
        })
    }

    pub fn pending(&self) -> usize {
        self.pending.load(Ordering::Relaxed)
    }

    pub fn max_pending(&self) -> usize {
        self.max_pending
    }
}

/// RAII guard that decrements the pending counter on drop.
/// Owns its references so it can be moved into spawned tasks.
pub struct QueueGuard {
    #[allow(dead_code)]
    _permit: tokio::sync::OwnedSemaphorePermit,
    pending: Arc<AtomicUsize>,
}

impl Drop for QueueGuard {
    fn drop(&mut self) {
        self.pending.fetch_sub(1, Ordering::Relaxed);
    }
}

#[derive(Serialize)]
struct ModelObject {
    id: String,
    object: String,
    owned_by: String,
}

#[derive(Serialize)]
struct ModelsResponse {
    object: String,
    data: Vec<ModelObject>,
}

pub async fn list_models<TG, IG>(
    _state: web::Data<Arc<RwLock<Master<TG, IG>>>>,
) -> HttpResponse
where
    TG: TextGenerator + Send + Sync + 'static,
    IG: ImageGenerator + Send + Sync + 'static,
{
    let response = ModelsResponse {
        object: "list".to_string(),
        data: vec![ModelObject {
            id: TG::MODEL_NAME.to_string(),
            object: "model".to_string(),
            owned_by: "cake".to_string(),
        }],
    };
    HttpResponse::Ok().json(response)
}

pub async fn list_models_video<TG, VG>(
    _state: web::Data<Arc<RwLock<VideoMaster<TG, VG>>>>,
) -> HttpResponse
where
    TG: TextGenerator + Send + Sync + 'static,
    VG: VideoGenerator + Send + Sync + 'static,
{
    let response = ModelsResponse {
        object: "list".to_string(),
        data: vec![ModelObject {
            id: VG::MODEL_NAME.to_string(),
            object: "model".to_string(),
            owned_by: "cake".to_string(),
        }],
    };
    HttpResponse::Ok().json(response)
}

async fn not_found() -> actix_web::Result<HttpResponse> {
    Ok(HttpResponse::NotFound().body("nope"))
}

/// GET /v1/status — queue depth and server health.
async fn status(queue: web::Data<Arc<RequestQueue>>) -> HttpResponse {
    HttpResponse::Ok().json(serde_json::json!({
        "status": "ok",
        "queue": {
            "pending": queue.pending(),
            "max_pending": queue.max_pending(),
        }
    }))
}

/// Maximum concurrent pending requests before returning 503.
const MAX_PENDING_REQUESTS: usize = 8;

pub(crate) async fn start<TG, IG>(master: Master<TG, IG>) -> anyhow::Result<()>
where
    TG: TextGenerator + Send + Sync + 'static,
    IG: ImageGenerator + Send + Sync + 'static,
{
    let address = master.ctx.args.api.as_ref().unwrap().to_string();

    log::info!("starting api on http://{} (max_pending={}) ...", &address, MAX_PENDING_REQUESTS);

    let state = Arc::new(RwLock::new(master));
    let queue = Arc::new(RequestQueue::new(MAX_PENDING_REQUESTS));

    HttpServer::new(
        move || {
            App::new()
                .app_data(web::Data::new(state.clone()))
                .app_data(web::Data::new(queue.clone()))
                .route(
                    "/v1/chat/completions",
                    web::post().to(generate_text::<TG, IG>),
                )
                .route(
                    "/api/v1/chat/completions",
                    web::post().to(generate_text::<TG, IG>),
                )
                .route("/v1/models", web::get().to(list_models::<TG, IG>))
                .route("/v1/status", web::get().to(status))
                .route("/api/v1/image", web::post().to(generate_image::<TG, IG>))
                .route("/api/v1/topology", web::get().to(ui::topology::<TG, IG>))
                .route("/", web::get().to(ui::index::<TG, IG>))
                .default_service(web::route().to(not_found))
        }, //.wrap(actix_web::middleware::Logger::default()))
    )
    .bind(&address)
    .map_err(|e| anyhow!(e))?
    .run()
    .await
    .map_err(|e| anyhow!(e))
}

pub(crate) async fn start_video<TG, VG>(master: VideoMaster<TG, VG>) -> anyhow::Result<()>
where
    TG: TextGenerator + Send + Sync + 'static,
    VG: VideoGenerator + Send + Sync + 'static,
{
    let address = master.ctx.args.api.as_ref().unwrap().to_string();

    log::info!("starting video api on http://{} ...", &address);

    let state = Arc::new(RwLock::new(master));

    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(state.clone()))
            .route("/v1/models", web::get().to(list_models_video::<TG, VG>))
            .route(
                "/api/v1/video",
                web::post().to(video::generate_video::<TG, VG>),
            )
            .default_service(web::route().to(not_found))
    })
    .bind(&address)
    .map_err(|e| anyhow!(e))?
    .run()
    .await
    .map_err(|e| anyhow!(e))
}
