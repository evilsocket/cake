use crate::cake::Master;
use crate::models::{Model, OutputModality};
use crate::ImageGenerationArgs;
use actix_web::{web, HttpRequest, HttpResponse, Responder};
use base64::engine::general_purpose;
use base64::Engine;
use image::{DynamicImage, ImageFormat};
use serde::{Deserialize, Serialize};
use std::io::Cursor;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;

/// Encode an RGB image buffer to PNG bytes.
fn encode_png(image: &image::ImageBuffer<image::Rgb<u8>, Vec<u8>>) -> Vec<u8> {
    let dynamic_image = DynamicImage::ImageRgb8(image.clone());
    let mut png_bytes = Vec::new();
    let mut cursor = Cursor::new(&mut png_bytes);
    dynamic_image
        .write_to(&mut cursor, ImageFormat::Png)
        .unwrap();
    png_bytes
}

// ── Legacy image endpoint (/api/v1/image) ──

#[derive(Deserialize)]
pub struct ImageRequest {
    pub image_args: ImageGenerationArgs,
    /// Response format: `"b64_json"` (default, backwards-compatible) or `"png"` (raw image/png).
    #[serde(default = "default_b64_json")]
    pub response_format: String,
}

fn default_b64_json() -> String {
    "b64_json".to_string()
}

#[derive(Serialize)]
struct ImageResponse {
    pub images: Vec<String>,
}

pub async fn generate_image<M: Model>(
    state: web::Data<Arc<RwLock<Master<M>>>>,
    req: HttpRequest,
    image_request: web::Json<ImageRequest>,
) -> impl Responder {
    let client = req
        .peer_addr()
        .map(|a| a.to_string())
        .unwrap_or_else(|| "unknown".to_string());

    log::info!("starting generating image for {} ...", &client);

    let mut master = state.write().await;

    if !master.model.as_ref().is_some_and(|m| m.output_modality() == OutputModality::Image) {
        return HttpResponse::NotFound()
            .json(serde_json::json!({"error": "No image model loaded"}));
    }

    let result_pngs: Arc<Mutex<Vec<Vec<u8>>>> = Arc::new(Mutex::new(Vec::new()));
    let result_pngs_cloned = Arc::clone(&result_pngs);

    if let Err(e) = master
        .generate_image(image_request.image_args.clone(), move |images| {
            let pngs: Vec<Vec<u8>> = images.iter().map(encode_png).collect();
            let mut locked = result_pngs_cloned.lock().expect("Error acquiring lock");
            locked.extend(pngs);
        })
        .await
    {
        return HttpResponse::InternalServerError()
            .json(serde_json::json!({"error": format!("{e}")}));
    }

    let locked = result_pngs.lock().expect("Error acquiring lock");

    if image_request.response_format == "png" {
        // Return raw PNG bytes for the first image
        if let Some(png) = locked.first() {
            HttpResponse::Ok()
                .content_type("image/png")
                .body(png.clone())
        } else {
            HttpResponse::InternalServerError()
                .json(serde_json::json!({"error": "No image generated"}))
        }
    } else {
        // Default: b64_json (backwards-compatible)
        let images: Vec<String> = locked
            .iter()
            .map(|png| general_purpose::STANDARD.encode(png))
            .collect();
        HttpResponse::Ok().json(ImageResponse { images })
    }
}

// ── OpenAI-compatible image endpoint (/v1/images/generations) ──

#[derive(Deserialize)]
#[allow(dead_code)]
pub struct OpenAIImageRequest {
    pub prompt: String,
    #[serde(default = "default_n")]
    pub n: usize,
    #[serde(default)]
    pub size: Option<String>,
    /// `"png"` (default, raw image/png), `"b64_json"` (OpenAI-compatible JSON wrapper).
    #[serde(default = "default_png")]
    pub response_format: String,
}

fn default_n() -> usize {
    1
}

fn default_png() -> String {
    "png".to_string()
}

#[derive(Serialize)]
struct OpenAIImageData {
    b64_json: String,
}

#[derive(Serialize)]
struct OpenAIImageResponse {
    created: u64,
    data: Vec<OpenAIImageData>,
}

pub async fn generate_image_openai<M: Model>(
    state: web::Data<Arc<RwLock<Master<M>>>>,
    req: HttpRequest,
    body: web::Json<OpenAIImageRequest>,
) -> impl Responder {
    let client = req
        .peer_addr()
        .map(|a| a.to_string())
        .unwrap_or_else(|| "unknown".to_string());

    log::info!("starting OpenAI image generation for {} ...", &client);

    let mut master = state.write().await;

    if !master.model.as_ref().is_some_and(|m| m.output_modality() == OutputModality::Image) {
        return HttpResponse::NotFound()
            .json(serde_json::json!({"error": "No image model loaded"}));
    }

    let args = ImageGenerationArgs::from_prompt(&body.prompt);

    let result_pngs: Arc<Mutex<Vec<Vec<u8>>>> = Arc::new(Mutex::new(Vec::new()));
    let result_pngs_cloned = Arc::clone(&result_pngs);

    if let Err(e) = master
        .generate_image(args, move |images| {
            let pngs: Vec<Vec<u8>> = images.iter().map(encode_png).collect();
            let mut locked = result_pngs_cloned.lock().expect("Error acquiring lock");
            locked.extend(pngs);
        })
        .await
    {
        return HttpResponse::InternalServerError()
            .json(serde_json::json!({"error": format!("{e}")}));
    }

    let locked = result_pngs.lock().expect("Error acquiring lock");

    if body.response_format == "b64_json" {
        // OpenAI-compatible JSON envelope
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let data: Vec<OpenAIImageData> = locked
            .iter()
            .map(|png| OpenAIImageData {
                b64_json: general_purpose::STANDARD.encode(png),
            })
            .collect();
        HttpResponse::Ok().json(OpenAIImageResponse { created, data })
    } else {
        // Default: raw PNG bytes
        if let Some(png) = locked.first() {
            HttpResponse::Ok()
                .content_type("image/png")
                .body(png.clone())
        } else {
            HttpResponse::InternalServerError()
                .json(serde_json::json!({"error": "No image generated"}))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_image_request_deserialization_full() {
        let json =
            r#"{"prompt": "A cat", "n": 2, "size": "512x512", "response_format": "b64_json"}"#;
        let req: OpenAIImageRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.prompt, "A cat");
        assert_eq!(req.n, 2);
        assert_eq!(req.size.as_deref(), Some("512x512"));
        assert_eq!(req.response_format, "b64_json");
    }

    #[test]
    fn test_openai_image_request_default_format_is_png() {
        let json = r#"{"prompt": "A dog"}"#;
        let req: OpenAIImageRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.response_format, "png");
    }

    #[test]
    fn test_legacy_image_request_default_format_is_b64() {
        let json = r#"{"image_args": {"sd-image-prompt": "test"}}"#;
        let req: ImageRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.response_format, "b64_json");
    }

    #[test]
    fn test_openai_image_response_serialization() {
        let resp = OpenAIImageResponse {
            created: 1234567890,
            data: vec![OpenAIImageData {
                b64_json: "abc123".into(),
            }],
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["created"], 1234567890);
        assert_eq!(json["data"][0]["b64_json"], "abc123");
    }
}
