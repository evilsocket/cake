//! Gemma 3 Instruct chat history encoder.
//!
//! Gemma 3 IT does NOT use a separate `<start_of_turn>system` role.
//! The system prompt is prepended to the FIRST user turn, per the official chat template:
//!
//!   <bos><start_of_turn>user
//!   {system_prompt (if any)}
//!   {user_message}<end_of_turn>
//!   <start_of_turn>model
//!   {model_response}<end_of_turn>
//!   ...
//!   <start_of_turn>model
//!   (prime generation)

use crate::models::chat::{Message, MessageRole};

#[derive(Debug, Default)]
pub struct Gemma3History {
    system: String,
    turns: Vec<Message>,
}

impl Gemma3History {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, msg: Message) {
        if msg.role == MessageRole::System {
            self.system = msg.content.trim().to_string();
        } else {
            self.turns.push(msg);
        }
    }

    pub fn clear(&mut self) {
        self.system.clear();
        self.turns.clear();
    }

    pub fn encode_dialog_to_prompt(&self) -> String {
        let mut out = String::new();
        // Include <bos> explicitly: our tokenizer is called with add_special_tokens=false
        // (no post-processor), so we must prepend it here. It encodes to token ID 2.
        out.push_str("<bos>");

        let mut first_user = true;
        for msg in &self.turns {
            let content = msg.content.trim();
            if content.is_empty() {
                continue;
            }
            match msg.role {
                MessageRole::User => {
                    out.push_str("<start_of_turn>user\n");
                    // Prepend system prompt to the first user turn only
                    if first_user && !self.system.is_empty() {
                        out.push_str(&self.system);
                        out.push('\n');
                        first_user = false;
                    }
                    out.push_str(content);
                    out.push_str("<end_of_turn>\n");
                }
                MessageRole::Assistant => {
                    out.push_str("<start_of_turn>model\n");
                    out.push_str(content);
                    out.push_str("<end_of_turn>\n");
                }
                MessageRole::System => unreachable!("system msgs are stored separately"),
            }
        }
        // Prime the model turn
        out.push_str("<start_of_turn>model\n");
        out
    }
}
