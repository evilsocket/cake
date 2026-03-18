pub(crate) mod audio;
pub(crate) mod image;
pub mod text;
mod ui;

#[cfg(test)]
mod test_helpers;
#[cfg(test)]
mod integration_tests;

use std::sync::Arc;

use actix_web::web;
use actix_web::App;
use actix_web::HttpResponse;
use actix_web::HttpServer;
use serde::Serialize;
use tokio::sync::RwLock;

use crate::models::{AudioGenerator, ImageGenerator, TextGenerator};

use audio::*;
use image::*;
use text::*;

use super::Master;

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

pub async fn list_models<TG, IG, AG>(
    state: web::Data<Arc<RwLock<Master<TG, IG, AG>>>>,
) -> HttpResponse
where
    TG: TextGenerator + Send + Sync + 'static,
    IG: ImageGenerator + Send + Sync + 'static,
    AG: AudioGenerator + Send + Sync + 'static,
{
    let master = state.read().await;
    let mut models = Vec::new();

    if master.llm_model.is_some() {
        models.push(ModelObject {
            id: TG::MODEL_NAME.to_string(),
            object: "model".to_string(),
            owned_by: "cake".to_string(),
        });
    }
    if master.sd_model.is_some() {
        models.push(ModelObject {
            id: IG::MODEL_NAME.to_string(),
            object: "model".to_string(),
            owned_by: "cake".to_string(),
        });
    }
    if master.audio_model.is_some() {
        models.push(ModelObject {
            id: AG::MODEL_NAME.to_string(),
            object: "model".to_string(),
            owned_by: "cake".to_string(),
        });
    }

    let response = ModelsResponse {
        object: "list".to_string(),
        data: models,
    };
    HttpResponse::Ok().json(response)
}

async fn not_found() -> actix_web::Result<HttpResponse> {
    Ok(HttpResponse::NotFound().body("nope"))
}

pub(crate) async fn start<TG, IG, AG>(master: Master<TG, IG, AG>) -> anyhow::Result<()>
where
    TG: TextGenerator + Send + Sync + 'static,
    IG: ImageGenerator + Send + Sync + 'static,
    AG: AudioGenerator + Send + Sync + 'static,
{
    let address = master.ctx.args.api.as_ref().unwrap().to_string();

    log::info!("starting api on http://{} ...", &address);

    let state = Arc::new(RwLock::new(master));

    HttpServer::new(
        move || {
            App::new()
                .app_data(web::Data::new(state.clone()))
                // Text / Chat
                .route(
                    "/v1/chat/completions",
                    web::post().to(generate_text::<TG, IG, AG>),
                )
                .route(
                    "/api/v1/chat/completions",
                    web::post().to(generate_text::<TG, IG, AG>),
                )
                // Models
                .route("/v1/models", web::get().to(list_models::<TG, IG, AG>))
                // Audio / TTS
                .route(
                    "/v1/audio/speech",
                    web::post().to(generate_speech::<TG, IG, AG>),
                )
                // Image (OpenAI-compatible)
                .route(
                    "/v1/images/generations",
                    web::post().to(generate_image_openai::<TG, IG, AG>),
                )
                // Image (legacy)
                .route(
                    "/api/v1/image",
                    web::post().to(generate_image::<TG, IG, AG>),
                )
                // Topology & UI
                .route(
                    "/api/v1/topology",
                    web::get().to(ui::topology::<TG, IG, AG>),
                )
                .route("/", web::get().to(ui::index::<TG, IG, AG>))
                .default_service(web::route().to(not_found))
        }, //.wrap(actix_web::middleware::Logger::default()))
    )
    .bind(&address)
    .map_err(|e| anyhow!(e))?
    .run()
    .await
    .map_err(|e| anyhow!(e))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── ModelObject / ModelsResponse serialization ──────────────

    #[test]
    fn test_model_object_serialization() {
        let obj = ModelObject {
            id: "test-model-v1".into(),
            object: "model".into(),
            owned_by: "cake".into(),
        };
        let json = serde_json::to_value(&obj).unwrap();
        assert_eq!(json["id"], "test-model-v1");
        assert_eq!(json["object"], "model");
        assert_eq!(json["owned_by"], "cake");
    }

    #[test]
    fn test_models_response_serialization() {
        let resp = ModelsResponse {
            object: "list".into(),
            data: vec![
                ModelObject {
                    id: "model-a".into(),
                    object: "model".into(),
                    owned_by: "cake".into(),
                },
                ModelObject {
                    id: "model-b".into(),
                    object: "model".into(),
                    owned_by: "cake".into(),
                },
            ],
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["object"], "list");
        assert_eq!(json["data"].as_array().unwrap().len(), 2);
        assert_eq!(json["data"][0]["id"], "model-a");
        assert_eq!(json["data"][1]["id"], "model-b");
    }

    #[test]
    fn test_models_response_empty_data() {
        let resp = ModelsResponse {
            object: "list".into(),
            data: vec![],
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["data"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn test_models_response_openai_compatible_structure() {
        // Verify the response matches OpenAI /v1/models format
        let resp = ModelsResponse {
            object: "list".into(),
            data: vec![ModelObject {
                id: "Qwen/Qwen3.5-0.8B".into(),
                object: "model".into(),
                owned_by: "cake".into(),
            }],
        };
        let json_str = serde_json::to_string(&resp).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        // OpenAI spec requires "object": "list" at top level
        assert_eq!(parsed["object"], "list");
        // Each model entry must have "object": "model"
        assert_eq!(parsed["data"][0]["object"], "model");
    }
}
