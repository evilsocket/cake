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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_stored_separately() {
        let mut h = Gemma3History::new();
        h.push(Message::system("You are helpful.".into()));
        h.push(Message::user("Hello".into()));
        // System message should not appear as a separate turn
        assert_eq!(h.turns.len(), 1);
        assert_eq!(h.system, "You are helpful.");
    }

    #[test]
    fn test_system_prepended_to_first_user() {
        let mut h = Gemma3History::new();
        h.push(Message::system("Be brief.".into()));
        h.push(Message::user("Hi".into()));

        let prompt = h.encode_dialog_to_prompt();
        assert!(prompt.starts_with("<bos>"));
        // System prepended before user content in first turn
        assert!(prompt.contains("<start_of_turn>user\nBe brief.\nHi<end_of_turn>"));
        assert!(prompt.ends_with("<start_of_turn>model\n"));
    }

    #[test]
    fn test_no_system_prompt() {
        let mut h = Gemma3History::new();
        h.push(Message::user("Hello".into()));

        let prompt = h.encode_dialog_to_prompt();
        assert!(prompt.contains("<start_of_turn>user\nHello<end_of_turn>"));
        assert!(prompt.ends_with("<start_of_turn>model\n"));
    }

    #[test]
    fn test_multi_turn() {
        let mut h = Gemma3History::new();
        h.push(Message::user("Hi".into()));
        h.push(Message::assistant("Hello!".into()));
        h.push(Message::user("How are you?".into()));

        let prompt = h.encode_dialog_to_prompt();
        assert!(prompt.contains("<start_of_turn>model\nHello!<end_of_turn>"));
        assert!(prompt.contains("<start_of_turn>user\nHow are you?<end_of_turn>"));
    }

    #[test]
    fn test_clear() {
        let mut h = Gemma3History::new();
        h.push(Message::system("sys".into()));
        h.push(Message::user("usr".into()));
        h.clear();
        assert!(h.system.is_empty());
        assert!(h.turns.is_empty());
    }
}
