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

use crate::models::Model;

use audio::*;
use image::*;
use text::*;

use super::master::Master;

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

pub async fn list_models<M: Model>(
    state: web::Data<Arc<RwLock<Master<M>>>>,
) -> HttpResponse {
    let master = state.read().await;
    let mut models = Vec::new();

    if master.model.is_some() {
        models.push(ModelObject {
            id: M::MODEL_NAME.to_string(),
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

pub(crate) async fn start<M: Model>(master: Master<M>) -> anyhow::Result<()> {
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
                    web::post().to(generate_text::<M>),
                )
                .route(
                    "/api/v1/chat/completions",
                    web::post().to(generate_text::<M>),
                )
                // Models
                .route("/v1/models", web::get().to(list_models::<M>))
                // Audio / TTS
                .route(
                    "/v1/audio/speech",
                    web::post().to(generate_speech::<M>),
                )
                // Image (OpenAI-compatible)
                .route(
                    "/v1/images/generations",
                    web::post().to(generate_image_openai::<M>),
                )
                // Image (legacy)
                .route(
                    "/api/v1/image",
                    web::post().to(generate_image::<M>),
                )
                // Topology & UI
                .route(
                    "/api/v1/topology",
                    web::get().to(ui::topology::<M>),
                )
                .route("/", web::get().to(ui::index::<M>))
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
                id: "evilsocket/Qwen3.5-0.8B".into(),
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
