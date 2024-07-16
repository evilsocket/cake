use std::io::Write;
use std::sync::Arc;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use actix_web::web;
use actix_web::App;
use actix_web::HttpRequest;
use actix_web::HttpResponse;
use actix_web::HttpServer;
use actix_web::Responder;
use serde::Deserialize;
use serde::Serialize;
use tokio::sync::RwLock;

use crate::models::chat::Message;
use crate::models::Generator;

use super::Master;

#[derive(Deserialize)]
struct Request {
    pub messages: Vec<Message>,
}

#[derive(Serialize)]
struct Choice {
    pub index: usize,
    pub message: Message,
}

#[derive(Serialize)]
struct Response {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
}

impl Response {
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

async fn chat<G>(
    state: web::Data<Arc<RwLock<Master<G>>>>,
    req: HttpRequest,
    messages: web::Json<Request>,
) -> impl Responder
where
    G: Generator + Send + Sync + 'static,
{
    let client = req.peer_addr().unwrap();

    log::info!("starting chat for {} ...", &client);

    let mut master = state.write().await;

    master.model.clear_history().unwrap();

    for message in messages.0.messages {
        master.model.add_message(message).unwrap();
    }

    let mut resp = String::new();

    // just run one generation to stdout
    master
        .generate(|data| {
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

    let response = Response::from_assistant_response(G::MODEL_NAME.to_string(), resp);

    HttpResponse::Ok().json(response)
}

async fn not_found() -> actix_web::Result<HttpResponse> {
    Ok(HttpResponse::NotFound().body("nope"))
}

pub(crate) async fn start<G>(master: Master<G>) -> anyhow::Result<()>
where
    G: Generator + Send + Sync + 'static,
{
    let address = master.ctx.args.api.as_ref().unwrap().to_string();

    log::info!("starting api on http://{} ...", &address);

    let state = Arc::new(RwLock::new(master));

    HttpServer::new(
        move || {
            App::new()
                .app_data(web::Data::new(state.clone()))
                .route("/api/v1/chat/completions", web::post().to(chat::<G>))
                .default_service(web::route().to(not_found))
        }, //.wrap(actix_web::middleware::Logger::default()))
    )
    .bind(&address)
    .map_err(|e| anyhow!(e))?
    .run()
    .await
    .map_err(|e| anyhow!(e))
}
