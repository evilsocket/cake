use crate::cake::Master;
use crate::models::chat::Message;
use crate::models::{ImageGenerator, TextGenerator};
use actix_web::{web, HttpRequest, HttpResponse, Responder};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;

#[derive(Deserialize)]
pub struct ChatRequest {
    pub messages: Vec<Message>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f64>,
}

#[derive(Serialize)]
struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Serialize)]
struct Choice {
    pub index: usize,
    pub message: Message,
    pub finish_reason: String,
}

#[derive(Serialize)]
struct ChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

impl ChatResponse {
    pub fn new(model: String, message: String, prompt_tokens: usize, completion_tokens: usize, finish_reason: String) -> Self {
        let id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
        let object = String::from("chat.completion");
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let choices = vec![Choice {
            index: 0,
            message: Message::assistant(message),
            finish_reason,
        }];

        Self {
            id,
            object,
            created,
            model,
            choices,
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
        }
    }
}

// SSE streaming types
#[derive(Serialize)]
struct StreamChoice {
    pub index: usize,
    pub delta: StreamDelta,
    pub finish_reason: Option<String>,
}

#[derive(Serialize)]
struct StreamDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

#[derive(Serialize)]
struct StreamResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<StreamChoice>,
}

pub async fn generate_text<TG, IG>(
    state: web::Data<Arc<RwLock<Master<TG, IG>>>>,
    req: HttpRequest,
    body: web::Json<ChatRequest>,
) -> impl Responder
where
    TG: TextGenerator + Send + Sync + 'static,
    IG: ImageGenerator + Send + Sync + 'static,
{
    let client = req
        .peer_addr()
        .map(|a| a.to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let stream = body.0.stream.unwrap_or(false);

    log::info!("starting chat for {} (stream={}) ...", &client, stream);

    if stream {
        generate_text_stream(state, body.0).await
    } else {
        generate_text_blocking(state, body.0).await
    }
}

async fn generate_text_blocking<TG, IG>(
    state: web::Data<Arc<RwLock<Master<TG, IG>>>>,
    request: ChatRequest,
) -> HttpResponse
where
    TG: TextGenerator + Send + Sync + 'static,
    IG: ImageGenerator + Send + Sync + 'static,
{
    let mut master = state.write().await;

    if let Err(e) = master.reset() {
        return HttpResponse::InternalServerError().json(serde_json::json!({"error": format!("{e}")}));
    }

    let num_messages = request.messages.len();
    let llm_model = master.llm_model.as_mut().expect("LLM model not found");
    for message in request.messages {
        if let Err(e) = llm_model.add_message(message) {
            return HttpResponse::InternalServerError().json(serde_json::json!({"error": format!("{e}")}));
        }
    }

    let mut resp = String::new();
    let mut finish_reason = "length".to_string();

    let gen_result = master
        .generate_text(|data| {
            if data.is_empty() {
                finish_reason = "stop".to_string();
            } else {
                resp += data;
                print!("{data}");
            }
            let _ = std::io::stdout().flush();
        })
        .await;

    println!();

    let completion_tokens = master
        .llm_model
        .as_ref()
        .expect("LLM model not found")
        .generated_tokens();

    let _ = master.goodbye().await;

    if let Err(e) = gen_result {
        return HttpResponse::InternalServerError().json(serde_json::json!({"error": format!("{e}")}));
    }

    let response = ChatResponse::new(
        TG::MODEL_NAME.to_string(),
        resp,
        num_messages,
        completion_tokens,
        finish_reason,
    );

    HttpResponse::Ok().json(response)
}

async fn generate_text_stream<TG, IG>(
    state: web::Data<Arc<RwLock<Master<TG, IG>>>>,
    request: ChatRequest,
) -> HttpResponse
where
    TG: TextGenerator + Send + Sync + 'static,
    IG: ImageGenerator + Send + Sync + 'static,
{
    let id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let model = TG::MODEL_NAME.to_string();

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Option<String>>();

    let state_clone = state.clone();
    tokio::spawn(async move {
        let mut master = state_clone.write().await;

        if let Err(e) = master.reset() {
            log::error!("reset error: {e}");
            let _ = tx.send(None);
            return;
        }

        let llm_model = master.llm_model.as_mut().expect("LLM model not found");
        for message in request.messages {
            if let Err(e) = llm_model.add_message(message) {
                log::error!("add_message error: {e}");
                let _ = tx.send(None);
                return;
            }
        }

        if let Err(e) = master
            .generate_text(|data| {
                if data.is_empty() {
                    let _ = tx.send(None);
                } else {
                    let _ = tx.send(Some(data.to_string()));
                }
            })
            .await
        {
            log::error!("generate_text error: {e}");
            let _ = tx.send(None);
        }

        let _ = master.goodbye().await;
    });

    let stream = async_stream::stream! {
        // Send initial role chunk
        let initial = StreamResponse {
            id: id.clone(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model.clone(),
            choices: vec![StreamChoice {
                index: 0,
                delta: StreamDelta {
                    role: Some("assistant".to_string()),
                    content: None,
                },
                finish_reason: None,
            }],
        };
        yield Ok::<_, actix_web::Error>(
            web::Bytes::from(format!("data: {}\n\n", serde_json::to_string(&initial).unwrap()))
        );

        // Stream content chunks
        while let Some(msg) = rx.recv().await {
            match msg {
                Some(content) => {
                    let chunk = StreamResponse {
                        id: id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: model.clone(),
                        choices: vec![StreamChoice {
                            index: 0,
                            delta: StreamDelta {
                                role: None,
                                content: Some(content),
                            },
                            finish_reason: None,
                        }],
                    };
                    yield Ok(web::Bytes::from(format!("data: {}\n\n", serde_json::to_string(&chunk).unwrap())));
                }
                None => {
                    // Final chunk with finish_reason
                    let done = StreamResponse {
                        id: id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: model.clone(),
                        choices: vec![StreamChoice {
                            index: 0,
                            delta: StreamDelta {
                                role: None,
                                content: None,
                            },
                            finish_reason: Some("stop".to_string()),
                        }],
                    };
                    yield Ok(web::Bytes::from(format!("data: {}\n\n", serde_json::to_string(&done).unwrap())));
                    yield Ok(web::Bytes::from("data: [DONE]\n\n"));
                    break;
                }
            }
        }
    };

    HttpResponse::Ok()
        .content_type("text/event-stream")
        .insert_header(("Cache-Control", "no-cache"))
        .insert_header(("Connection", "keep-alive"))
        .streaming(stream)
}
