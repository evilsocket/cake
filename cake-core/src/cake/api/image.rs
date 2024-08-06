use std::io::Cursor;
use std::sync::Arc;
use actix_web::{HttpRequest, HttpResponse, Responder, web};
use base64::Engine;
use base64::engine::general_purpose;
use image::{DynamicImage, ImageFormat};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use crate::cake::Master;
use crate::ImageGenerationArgs;
use crate::models::{TextGenerator};
use crate::models::ImageGenerator;

#[derive(Deserialize)]
pub struct ImageRequest {
    pub image_args: ImageGenerationArgs
}

#[derive(Serialize)]
struct ImageResponse {
    pub images: Vec<String>,
}

pub async fn generate_image<TG, IG>(
    state: web::Data<Arc<RwLock<Master<TG, IG>>>>,
    req: HttpRequest,
    image_request: web::Json<ImageRequest>,
) -> impl Responder
where
    TG: TextGenerator + Send + Sync + 'static,
    IG: ImageGenerator + Send + Sync + 'static,
{
    let client = req.peer_addr().unwrap();

    log::info!("starting generating image for {} ...", &client);

    let mut master = state.write().await;

    let images = master.generate_image(image_request.image_args.clone()).await.expect("Error generating images using API");

    let base64_images: Vec<String> = images
        .iter()
        .map(|image| {

            let dynamic_image = DynamicImage::ImageRgb8(image.clone());
            let mut png_bytes = Vec::new();
            let mut cursor = Cursor::new(&mut png_bytes);
            dynamic_image.write_to(&mut cursor, ImageFormat::Png).unwrap();
            general_purpose::STANDARD.encode(png_bytes)
        })
        .collect();

    let response = ImageResponse{
        images: base64_images
    };

    HttpResponse::Ok().json(response)
}