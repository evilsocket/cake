//! Integration tests for the Cake distributed LLM inference framework.
//!
//! These tests validate that a model integration works correctly end-to-end:
//! loading, token generation, chat coherence, state management, and API compatibility.
//!
//! LLaMA tests require CAKE_TEST_MODEL (default: ./cake-data/Llama-3.2-1B-Instruct/).
//! Qwen2 tests require CAKE_TEST_QWEN2_MODEL env var (skipped if not set).
//!
//! Run with: cargo test --test integration -- --test-threads=1

use std::env;
use std::sync::Arc;

use cake_core::cake::{Context, Master, Mode};
use cake_core::models::chat::Message;
use cake_core::models::{Generator, TextGenerator};
use cake_core::{Args, ModelType, TextModelArch};

use tokio::sync::Mutex;

// =============================================================================
// Generic test helpers
// =============================================================================

fn test_args(model_path: &str, sample_len: usize, arch: TextModelArch) -> Args {
    Args {
        model: model_path.to_string(),
        mode: Mode::Master,
        model_type: ModelType::TextModel,
        text_model_arch: arch,
        sample_len,
        temperature: 0.0, // deterministic (argmax)
        seed: 42,
        cpu: false,
        system_prompt: "You are a helpful AI assistant.".into(),
        repeat_penalty: 1.1,
        repeat_last_n: 128,
        ..Default::default()
    }
}

/// Helper: run a single-turn chat and return the response text.
async fn chat<TG, IG>(master: &mut Master<TG, IG>, system: &str, user: &str) -> String
where
    TG: TextGenerator + Send + Sync + 'static,
    IG: cake_core::models::ImageGenerator + Send + Sync + 'static,
{
    master.reset().unwrap();

    let llm = master.llm_model.as_mut().expect("LLM model not found");
    llm.add_message(Message::system(system.into())).unwrap();
    llm.add_message(Message::user(user.into())).unwrap();

    let mut response = String::new();
    master
        .generate_text(None, |data| {
            if !data.is_empty() {
                response.push_str(data);
            }
        })
        .await
        .unwrap();

    master.goodbye().await.unwrap();
    response
}

/// Macro to generate the full test suite for a given model type.
/// This avoids duplicating all test bodies between LLaMA and Qwen2.
macro_rules! model_test_suite {
    (
        module_name: $mod_name:ident,
        text_model: $text_model:ty,
        image_model: $image_model:ty,
        model_name_const: $model_name:expr,
        env_var: $env_var:expr,
        default_path: $default_path:expr,
        arch: $arch:expr,
        skip_if_missing: $skip:expr,
    ) => {
        mod $mod_name {
            use super::*;
            use tokio::sync::OnceCell;

            type TestMaster = Master<$text_model, $image_model>;

            static MODEL: OnceCell<Arc<Mutex<TestMaster>>> = OnceCell::const_new();

            fn get_model_path() -> Option<String> {
                if $skip {
                    env::var($env_var).ok()
                } else {
                    Some(env::var($env_var).unwrap_or_else(|_| $default_path.into()))
                }
            }

            async fn get_or_load_model() -> Option<Arc<Mutex<TestMaster>>> {
                let path = get_model_path()?;
                Some(
                    MODEL
                        .get_or_init(|| async {
                            let args = test_args(&path, 100, $arch);
                            let ctx = Context::from_args(args).expect("Failed to create context");
                            let master =
                                TestMaster::new(ctx).await.expect("Failed to load model");
                            Arc::new(Mutex::new(master))
                        })
                        .await
                        .clone(),
                )
            }

            // =================================================================
            // Category 1: Model Loading
            // =================================================================

            #[tokio::test]
            async fn test_model_loads_successfully() {
                let Some(model) = get_or_load_model().await else {
                    return;
                };
                let master = model.lock().await;
                assert!(master.llm_model.is_some(), "LLM model should be loaded");
            }

            #[tokio::test]
            async fn test_model_has_valid_name() {
                if get_model_path().is_none() {
                    return;
                }
                assert!(
                    !<$text_model as Generator>::MODEL_NAME.is_empty(),
                    "MODEL_NAME must be non-empty"
                );
                assert_eq!(
                    <$text_model as Generator>::MODEL_NAME,
                    $model_name,
                    "MODEL_NAME should match expected value"
                );
            }

            #[tokio::test]
            async fn test_invalid_model_path_fails() {
                if get_model_path().is_none() {
                    return;
                }
                let args = test_args("/nonexistent/path/to/model", 10, $arch);
                let result = Context::from_args(args);
                assert!(result.is_err(), "Non-existent model path should return Err");
            }

            // =================================================================
            // Category 2: Token Generation
            // =================================================================

            #[tokio::test]
            async fn test_generates_tokens() {
                let Some(model) = get_or_load_model().await else {
                    return;
                };
                let mut master = model.lock().await;
                let response =
                    chat(&mut master, "You are a helpful assistant.", "Hello").await;
                assert!(
                    !response.is_empty(),
                    "Model must generate at least one token"
                );
            }

            #[tokio::test]
            async fn test_tokens_have_text() {
                let Some(model) = get_or_load_model().await else {
                    return;
                };
                let mut master = model.lock().await;
                master.reset().unwrap();

                let llm = master.llm_model.as_mut().unwrap();
                llm.add_message(Message::system(
                    "You are a helpful assistant.".into(),
                ))
                .unwrap();
                llm.add_message(Message::user("Count to five.".into()))
                    .unwrap();

                let mut had_text = false;
                for index in 0..20 {
                    let token = llm.next_token(index).await.unwrap();
                    if token.is_end_of_stream {
                        break;
                    }
                    assert!(
                        token.text.is_some(),
                        "Token {} should have decodable text",
                        token.id
                    );
                    had_text = true;
                }
                assert!(
                    had_text,
                    "Should have generated at least one token with text"
                );

                master.goodbye().await.unwrap();
            }

            #[tokio::test]
            async fn test_eos_terminates_generation() {
                let Some(model) = get_or_load_model().await else {
                    return;
                };
                let mut master = model.lock().await;
                master.reset().unwrap();

                let llm = master.llm_model.as_mut().unwrap();
                llm.add_message(Message::system(
                    "You are a helpful assistant. Answer concisely.".into(),
                ))
                .unwrap();
                llm.add_message(Message::user(
                    "What is 2+2? Answer with just the number.".into(),
                ))
                .unwrap();

                let mut received_eos = false;
                master
                    .generate_text(None, |data| {
                        if data.is_empty() {
                            received_eos = true;
                        }
                    })
                    .await
                    .unwrap();

                assert!(
                    received_eos,
                    "Should receive empty string signal (EOS) at end of generation"
                );

                master.goodbye().await.unwrap();
            }

            #[tokio::test]
            async fn test_generated_tokens_counter() {
                let Some(model) = get_or_load_model().await else {
                    return;
                };
                let mut master = model.lock().await;
                master.reset().unwrap();

                let llm = master.llm_model.as_mut().unwrap();
                assert_eq!(llm.generated_tokens(), 0, "Should start at 0 tokens");

                llm.add_message(Message::system(
                    "You are a helpful assistant.".into(),
                ))
                .unwrap();
                llm.add_message(Message::user("Hi".into())).unwrap();

                for i in 0..5 {
                    let token = llm.next_token(i).await.unwrap();
                    if token.is_end_of_stream {
                        break;
                    }
                }

                assert!(
                    llm.generated_tokens() > 0,
                    "Token counter must increase after generation"
                );

                master.goodbye().await.unwrap();
            }

            // =================================================================
            // Category 3: Chat Coherence
            // =================================================================

            #[tokio::test]
            async fn test_arithmetic() {
                let Some(model) = get_or_load_model().await else {
                    return;
                };
                let mut master = model.lock().await;
                let response = chat(
                    &mut master,
                    "You are a helpful assistant. Answer concisely.",
                    "What is 2+2?",
                )
                .await;

                assert!(
                    response.contains("4"),
                    "Expected '4' in response to '2+2', got: '{}'",
                    response
                );
            }

            #[tokio::test]
            async fn test_factual_knowledge() {
                let Some(model) = get_or_load_model().await else {
                    return;
                };
                let mut master = model.lock().await;
                let response = chat(
                    &mut master,
                    "You are a helpful assistant. Answer concisely.",
                    "What is the capital of France?",
                )
                .await;

                assert!(
                    response.to_lowercase().contains("paris"),
                    "Expected 'Paris' in response, got: '{}'",
                    response
                );
            }

            #[tokio::test]
            async fn test_not_gibberish() {
                let Some(model) = get_or_load_model().await else {
                    return;
                };
                let mut master = model.lock().await;
                let response = chat(
                    &mut master,
                    "You are a helpful assistant.",
                    "Explain what water is in one sentence.",
                )
                .await;

                // Must contain some common English words
                let lower = response.to_lowercase();
                let common_words = ["the", "is", "a", "water", "and", "of", "it"];
                let matches =
                    common_words.iter().filter(|w| lower.contains(**w)).count();
                assert!(
                    matches >= 2,
                    "Expected response to contain common English words, got: '{}'",
                    response
                );

                // Average word length should be reasonable (not random bytes)
                let words: Vec<&str> = response.split_whitespace().collect();
                if !words.is_empty() {
                    let avg_len: f64 = words.iter().map(|w| w.len() as f64).sum::<f64>()
                        / words.len() as f64;
                    assert!(
                        avg_len < 20.0,
                        "Average word length {:.1} is too high, suggesting gibberish",
                        avg_len
                    );
                    assert!(
                        avg_len > 1.0,
                        "Average word length {:.1} is too low",
                        avg_len
                    );
                }
            }

            // =================================================================
            // Category 4: State Management
            // =================================================================

            #[tokio::test]
            async fn test_reset_clears_state() {
                let Some(model) = get_or_load_model().await else {
                    return;
                };
                let mut master = model.lock().await;

                // Generate something first
                let _ =
                    chat(&mut master, "You are a helpful assistant.", "Hello").await;

                let tokens_before =
                    master.llm_model.as_ref().unwrap().generated_tokens();
                assert!(tokens_before > 0, "Should have generated tokens");

                // Reset
                master.reset().unwrap();
                assert_eq!(
                    master.llm_model.as_ref().unwrap().generated_tokens(),
                    0,
                    "generated_tokens should be 0 after reset"
                );
            }

            #[tokio::test]
            async fn test_independent_conversations() {
                let Some(model) = get_or_load_model().await else {
                    return;
                };
                let mut master = model.lock().await;

                // First conversation about math
                let response1 = chat(
                    &mut master,
                    "You are a helpful assistant.",
                    "What is 2+2?",
                )
                .await;
                assert!(response1.contains("4"));

                // Second conversation about geography (independent)
                let response2 = chat(
                    &mut master,
                    "You are a helpful assistant.",
                    "What is the capital of France?",
                )
                .await;

                assert!(
                    response2.to_lowercase().contains("paris"),
                    "Second conversation should be independent, got: '{}'",
                    response2
                );
            }

            #[tokio::test]
            async fn test_multi_turn_conversation() {
                let Some(model) = get_or_load_model().await else {
                    return;
                };
                let mut master = model.lock().await;
                master.reset().unwrap();

                let llm = master.llm_model.as_mut().unwrap();
                llm.add_message(Message::system(
                    "You are a helpful assistant.".into(),
                ))
                .unwrap();
                llm.add_message(Message::user(
                    "What is the largest planet?".into(),
                ))
                .unwrap();
                llm.add_message(Message::assistant(
                    "Jupiter is the largest planet.".into(),
                ))
                .unwrap();
                llm.add_message(Message::user(
                    "How many moons does it have?".into(),
                ))
                .unwrap();

                let mut response = String::new();
                master
                    .generate_text(None, |data| {
                        if !data.is_empty() {
                            response.push_str(data);
                        }
                    })
                    .await
                    .unwrap();

                assert!(
                    !response.is_empty(),
                    "Multi-turn conversation should produce a response"
                );

                master.goodbye().await.unwrap();
            }

            // =================================================================
            // Category 5: API Compatibility
            // =================================================================

            #[cfg(feature = "master")]
            mod api_tests {
                use super::*;
                use actix_web::{test, web, App};
                use cake_core::cake::api;
                use tokio::sync::RwLock;

                async fn load_model_for_api() -> Option<TestMaster> {
                    let path = get_model_path()?;
                    let args = test_args(&path, 50, $arch);
                    let ctx =
                        Context::from_args(args).expect("Failed to create context");
                    Some(
                        TestMaster::new(ctx)
                            .await
                            .expect("Failed to load model"),
                    )
                }

                #[actix_web::test]
                async fn test_api_blocking_response() {
                    let Some(master) = load_model_for_api().await else {
                        return;
                    };
                    let state = Arc::new(RwLock::new(master));

                    let app = test::init_service(
                        App::new()
                            .app_data(web::Data::new(state))
                            .route(
                                "/v1/chat/completions",
                                web::post().to(
                                    api::text::generate_text::<
                                        $text_model,
                                        $image_model,
                                    >,
                                ),
                            ),
                    )
                    .await;

                    let body = serde_json::json!({
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": "What is 2+2?"}
                        ],
                        "stream": false
                    });

                    let req = test::TestRequest::post()
                        .uri("/v1/chat/completions")
                        .set_json(&body)
                        .to_request();

                    let resp: serde_json::Value =
                        test::call_and_read_body_json(&app, req).await;

                    // Validate OpenAI response format
                    assert!(resp["id"]
                        .as_str()
                        .unwrap()
                        .starts_with("chatcmpl-"));
                    assert_eq!(resp["object"], "chat.completion");
                    assert!(resp["created"].as_u64().is_some());

                    // Validate choices
                    let choices = resp["choices"].as_array().unwrap();
                    assert_eq!(choices.len(), 1);
                    assert_eq!(choices[0]["index"], 0);
                    assert_eq!(choices[0]["message"]["role"], "assistant");
                    assert!(!choices[0]["message"]["content"]
                        .as_str()
                        .unwrap()
                        .is_empty());
                    assert!(["stop", "length"].contains(
                        &choices[0]["finish_reason"].as_str().unwrap()
                    ));

                    // Validate usage
                    assert!(
                        resp["usage"]["completion_tokens"].as_u64().unwrap() > 0
                    );
                    assert!(resp["usage"]["total_tokens"].as_u64().unwrap() > 0);
                }

                #[actix_web::test]
                async fn test_api_models_endpoint() {
                    let Some(master) = load_model_for_api().await else {
                        return;
                    };
                    let state = Arc::new(RwLock::new(master));

                    let app = test::init_service(
                        App::new()
                            .app_data(web::Data::new(state))
                            .route(
                                "/v1/models",
                                web::get().to(
                                    api::list_models::<$text_model, $image_model>,
                                ),
                            ),
                    )
                    .await;

                    let req =
                        test::TestRequest::get().uri("/v1/models").to_request();
                    let resp: serde_json::Value =
                        test::call_and_read_body_json(&app, req).await;

                    assert_eq!(resp["object"], "list");
                    let data = resp["data"].as_array().unwrap();
                    assert_eq!(data.len(), 1);
                    assert_eq!(data[0]["object"], "model");
                    assert!(!data[0]["id"].as_str().unwrap().is_empty());
                    assert_eq!(data[0]["owned_by"], "cake");
                }
            }
        }
    };
}

// =============================================================================
// LLaMA test suite
// =============================================================================

#[cfg(feature = "llama")]
model_test_suite! {
    module_name: llama_tests,
    text_model: cake_core::models::llama3::LLama,
    image_model: cake_core::models::sd::SD,
    model_name_const: "llama3",
    env_var: "CAKE_TEST_MODEL",
    default_path: "./cake-data/Llama-3.2-1B-Instruct/",
    arch: TextModelArch::Llama,
    skip_if_missing: false,
}

// =============================================================================
// Qwen2 test suite (same tests, different model)
// =============================================================================

#[cfg(feature = "qwen2")]
model_test_suite! {
    module_name: qwen2_tests,
    text_model: cake_core::models::qwen2::Qwen2,
    image_model: cake_core::models::sd::SD,
    model_name_const: "qwen2",
    env_var: "CAKE_TEST_QWEN2_MODEL",
    default_path: "",
    arch: TextModelArch::Qwen2,
    skip_if_missing: true,
}
