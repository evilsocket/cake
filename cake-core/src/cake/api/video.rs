use crate::cake::master::VideoMaster;
use crate::models::TextGenerator;
use crate::models::VideoGenerator;
use crate::ImageGenerationArgs;
use actix_web::{web, HttpRequest, HttpResponse, Responder};
use base64::engine::general_purpose;
use base64::Engine;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Deserialize)]
pub struct VideoRequest {
    pub image_args: ImageGenerationArgs,
    /// Output format: "avi" (binary) or "base64" (JSON with base64-encoded AVI).
    /// Default: "avi"
    #[serde(default = "default_format")]
    pub format: String,
    /// If true, also return individual frames as base64 PNGs alongside the video.
    #[serde(default)]
    pub include_frames: bool,
}

fn default_format() -> String {
    "avi".to_string()
}

#[derive(Serialize)]
struct VideoJsonResponse {
    /// Base64-encoded AVI data.
    pub video: String,
    /// Video format identifier.
    pub format: String,
    /// Number of frames.
    pub num_frames: usize,
    /// Frames per second.
    pub fps: usize,
    /// Frame width.
    pub width: u32,
    /// Frame height.
    pub height: u32,
    /// Duration in seconds.
    pub duration_secs: f64,
    /// Optional individual frames as base64 PNGs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frames: Option<Vec<String>>,
}

pub async fn generate_video<TG, VG>(
    state: web::Data<Arc<RwLock<VideoMaster<TG, VG>>>>,
    req: HttpRequest,
    video_request: web::Json<VideoRequest>,
) -> impl Responder
where
    TG: TextGenerator + Send + Sync + 'static,
    VG: VideoGenerator + Send + Sync + 'static,
{
    let client = req
        .peer_addr()
        .map(|a| a.to_string())
        .unwrap_or_else(|| "unknown".to_string());
    log::info!("starting video generation for {} ...", &client);

    let mut master = state.write().await;

    let video_output = match master.generate_video(video_request.image_args.clone()).await {
        Ok(v) => v,
        Err(e) => {
            log::error!("video generation failed: {}", e);
            return HttpResponse::InternalServerError()
                .json(serde_json::json!({"error": e.to_string()}));
        }
    };

    let avi_bytes = match video_output.to_avi() {
        Ok(b) => b,
        Err(e) => {
            log::error!("AVI encoding failed: {}", e);
            return HttpResponse::InternalServerError()
                .json(serde_json::json!({"error": e.to_string()}));
        }
    };

    match video_request.format.as_str() {
        "avi" | "binary" => {
            // Return raw AVI bytes
            HttpResponse::Ok()
                .content_type("video/x-msvideo")
                .append_header((
                    "Content-Disposition",
                    "attachment; filename=\"output.avi\"",
                ))
                .body(avi_bytes)
        }
        _ => {
            // Return JSON with base64-encoded video
            let frames = if video_request.include_frames {
                Some(encode_frames_as_png(&video_output))
            } else {
                None
            };

            let response = VideoJsonResponse {
                video: general_purpose::STANDARD.encode(&avi_bytes),
                format: "avi".to_string(),
                num_frames: video_output.num_frames(),
                fps: video_output.fps,
                width: video_output.width,
                height: video_output.height,
                duration_secs: video_output.duration_secs(),
                frames,
            };

            HttpResponse::Ok().json(response)
        }
    }
}

fn encode_frames_as_png(video: &crate::video::VideoOutput) -> Vec<String> {
    use image::{DynamicImage, ImageFormat};
    use std::io::Cursor;

    video
        .frames
        .iter()
        .map(|frame| {
            let dynamic_image = DynamicImage::ImageRgb8(frame.clone());
            let mut png_bytes = Vec::new();
            let mut cursor = Cursor::new(&mut png_bytes);
            dynamic_image
                .write_to(&mut cursor, ImageFormat::Png)
                .unwrap();
            general_purpose::STANDARD.encode(png_bytes)
        })
        .collect()
}
