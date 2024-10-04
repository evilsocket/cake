mod image;
mod text;

use std::sync::Arc;

use actix_web::web;
use actix_web::App;
use actix_web::HttpResponse;
use actix_web::HttpServer;
use tokio::sync::RwLock;

use crate::models::{ImageGenerator, TextGenerator};

use image::*;
use text::*;

use super::Master;

async fn not_found() -> actix_web::Result<HttpResponse> {
    Ok(HttpResponse::NotFound().body("nope"))
}

pub(crate) async fn start<TG, IG>(master: Master<TG, IG>) -> anyhow::Result<()>
where
    TG: TextGenerator + Send + Sync + 'static,
    IG: ImageGenerator + Send + Sync + 'static,
{
    let address = master.ctx.args.api.as_ref().unwrap().to_string();

    log::info!("starting api on http://{} ...", &address);

    let state = Arc::new(RwLock::new(master));

    HttpServer::new(
        move || {
            App::new()
                .app_data(web::Data::new(state.clone()))
                .route(
                    "/api/v1/chat/completions",
                    web::post().to(generate_text::<TG, IG>),
                )
                .route("/api/v1/image", web::post().to(generate_image::<TG, IG>))
                .default_service(web::route().to(not_found))
        }, //.wrap(actix_web::middleware::Logger::default()))
    )
    .bind(&address)
    .map_err(|e| anyhow!(e))?
    .run()
    .await
    .map_err(|e| anyhow!(e))
}
