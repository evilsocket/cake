use std::io::Write;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use actix_web::{HttpRequest, HttpResponse, Responder, web};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use crate::cake::Master;
use crate::models::chat::Message;
use crate::models::{ImageGenerator, TextGenerator};

#[derive(Deserialize)]
pub struct ChatRequest {
    pub messages: Vec<Message>,
}

#[derive(Serialize)]
struct Choice {
    pub index: usize,
    pub message: Message,
}

#[derive(Serialize)]
struct ChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
}

impl ChatResponse {
    pub fn from_assistant_response(model: String, message: String) -> Self {
        let id = uuid::Uuid::new_v4().to_string();
        let object = String::from("chat.completion");
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let choices = vec![Choice {
            index: 0,
            message: Message::assistant(message),
        }];

        Self {
            id,
            object,
            created,
            model,
            choices,
        }
    }
}

pub async fn generate_text<TG, IG>(
    state: web::Data<Arc<RwLock<Master<TG, IG>>>>,
    req: HttpRequest,
    messages: web::Json<ChatRequest>,
) -> impl Responder
where
    TG: TextGenerator + Send + Sync + 'static,
    IG: ImageGenerator + Send + Sync + 'static,
{
    let client = req.peer_addr().unwrap();

    log::info!("starting chat for {} ...", &client);

    let mut master = state.write().await;

    master.reset().unwrap();

    let llm_model = master.llm_model.as_mut().expect("LLM model not found");

    for message in messages.0.messages {
        llm_model.add_message(message).unwrap();
    }

    let mut resp = String::new();

    // just run one generation to stdout
    master
        .generate_text(|data| {
            resp += data;
            if data.is_empty() {
                println!();
            } else {
                print!("{data}")
            }
            std::io::stdout().flush().unwrap();
        })
        .await
        .unwrap();

    let response = ChatResponse::from_assistant_response(TG::MODEL_NAME.to_string(), resp);
    master.goodbye().await.unwrap();

    HttpResponse::Ok().json(response)
}