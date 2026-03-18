use crate::cake::Master;
use crate::models::{AudioGenerationArgs, Model, OutputModality};
use actix_web::{web, HttpRequest, HttpResponse, Responder};
use base64::engine::general_purpose;
use base64::Engine;
use serde::Deserialize;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Deserialize)]
#[allow(dead_code)]
pub struct AudioSpeechRequest {
    pub input: String,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub voice: Option<String>,
    /// Base64-encoded WAV bytes for voice cloning.
    #[serde(default)]
    pub voice_data: Option<String>,
    /// Path to voice prompt file on the server.
    #[serde(default)]
    pub voice_path: Option<String>,
    #[serde(default = "default_response_format")]
    pub response_format: String,
    #[serde(default = "default_cfg_scale")]
    pub cfg_scale: f32,
    #[serde(default = "default_max_frames")]
    pub max_frames: usize,
    #[serde(default = "default_diffusion_steps")]
    pub diffusion_steps: usize,
}

fn default_response_format() -> String {
    "wav".to_string()
}

fn default_cfg_scale() -> f32 {
    1.5
}

fn default_max_frames() -> usize {
    150
}

fn default_diffusion_steps() -> usize {
    10
}

pub async fn generate_speech<M: Model>(
    state: web::Data<Arc<RwLock<Master<M>>>>,
    req: HttpRequest,
    body: web::Json<AudioSpeechRequest>,
) -> impl Responder {
    let client = req
        .peer_addr()
        .map(|a| a.to_string())
        .unwrap_or_else(|| "unknown".to_string());

    log::info!("starting audio generation for {} ...", &client);

    let mut master = state.write().await;

    if !master.model.as_ref().is_some_and(|m| m.output_modality() == OutputModality::Audio) {
        return HttpResponse::NotFound()
            .json(serde_json::json!({"error": "No audio model loaded"}));
    }

    // Decode base64 voice data if provided
    let voice_data = match &body.voice_data {
        Some(b64) => match general_purpose::STANDARD.decode(b64) {
            Ok(bytes) => Some(bytes),
            Err(e) => {
                return HttpResponse::BadRequest()
                    .json(serde_json::json!({"error": format!("Invalid voice_data base64: {e}")}));
            }
        },
        None => None,
    };

    let args = AudioGenerationArgs {
        input: body.input.clone(),
        voice_data,
        voice_path: body.voice_path.clone(),
        cfg_scale: body.cfg_scale,
        max_frames: body.max_frames,
        diffusion_steps: body.diffusion_steps,
    };

    match master.generate_audio(&args).await {
        Ok(output) => {
            if body.response_format == "pcm" {
                HttpResponse::Ok()
                    .content_type("audio/pcm")
                    .body(output.to_pcm_bytes())
            } else {
                HttpResponse::Ok()
                    .content_type("audio/wav")
                    .body(output.to_wav_bytes())
            }
        }
        Err(e) => HttpResponse::InternalServerError()
            .json(serde_json::json!({"error": format!("{e}")})),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_speech_request_full() {
        let json = r#"{
            "input": "Hello world",
            "model": "vibevoice/VibeVoice-1.5B",
            "voice": "default",
            "voice_data": "AAAA",
            "response_format": "pcm",
            "cfg_scale": 2.0,
            "max_frames": 200,
            "diffusion_steps": 20
        }"#;
        let req: AudioSpeechRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.input, "Hello world");
        assert_eq!(req.model.as_deref(), Some("vibevoice/VibeVoice-1.5B"));
        assert_eq!(req.voice.as_deref(), Some("default"));
        assert!(req.voice_data.is_some());
        assert_eq!(req.response_format, "pcm");
        assert!((req.cfg_scale - 2.0).abs() < 1e-6);
        assert_eq!(req.max_frames, 200);
        assert_eq!(req.diffusion_steps, 20);
    }

    #[test]
    fn test_audio_speech_request_minimal() {
        let json = r#"{"input": "Test"}"#;
        let req: AudioSpeechRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.input, "Test");
        assert!(req.model.is_none());
        assert!(req.voice.is_none());
        assert!(req.voice_data.is_none());
        assert!(req.voice_path.is_none());
        assert_eq!(req.response_format, "wav");
        assert!((req.cfg_scale - 1.5).abs() < 1e-6);
        assert_eq!(req.max_frames, 150);
        assert_eq!(req.diffusion_steps, 10);
    }

    #[test]
    fn test_audio_speech_request_missing_input_fails() {
        let json = r#"{"model": "test"}"#;
        let result = serde_json::from_str::<AudioSpeechRequest>(json);
        assert!(result.is_err());
    }
}
