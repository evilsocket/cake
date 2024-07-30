use std::sync::Arc;
use actix_web::{HttpRequest, Responder, web};
use serde::Deserialize;
use tokio::sync::RwLock;
use crate::cake::Master;
use crate::models::{ImageGenerationParameters, TextGenerator};
use crate::models::ImageGenerator;

#[derive(Deserialize)]
struct ImageRequest {
    pub parameters: ImageGenerationParameters,
}

pub async fn generate_image<TG, IG>(
    state: web::Data<Arc<RwLock<Master<TG, IG>>>>,
    req: HttpRequest,
    generation_parameters: web::Json<ImageRequest>,
) -> impl Responder
where
    TG: TextGenerator + Send + Sync + 'static,
    IG: ImageGenerator + Send + Sync + 'static,
{
    let client = req.peer_addr().unwrap();

    log::info!("starting generating image for {} ...", &client);

    let mut master = state.write().await;

    master.reset().unwrap();
}