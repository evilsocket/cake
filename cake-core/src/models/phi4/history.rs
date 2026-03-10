//! Phi-3 / Phi-4 chat history encoder.
//!
//! Format:
//!   <|system|>\n{system}<|end|>\n<|user|>\n{user}<|end|>\n<|assistant|>\n{assistant}<|end|>\n...
//!   Then a trailing <|assistant|>\n to prime generation.

use crate::models::chat::Message;

#[derive(Debug, Default)]
pub struct Phi4History(Vec<Message>);

impl Phi4History {
    pub fn new() -> Self {
        Self(vec![])
    }

    pub fn push(&mut self, msg: Message) {
        self.0.push(msg);
    }

    pub fn clear(&mut self) {
        self.0.clear();
    }

    pub fn encode_dialog_to_prompt(&self) -> String {
        let mut out = String::new();
        for msg in &self.0 {
            out.push_str(&format!("<|{}|>\n{}<|end|>\n", msg.role, msg.content.trim()));
        }
        // Prime the assistant turn
        out.push_str("<|assistant|>\n");
        out
    }
}
