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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phi4_single_user() {
        let mut h = Phi4History::new();
        h.push(Message::user("Hello".into()));

        let prompt = h.encode_dialog_to_prompt();
        assert_eq!(prompt, "<|user|>\nHello<|end|>\n<|assistant|>\n");
    }

    #[test]
    fn test_phi4_system_user() {
        let mut h = Phi4History::new();
        h.push(Message::system("You are helpful.".into()));
        h.push(Message::user("Hi".into()));

        let prompt = h.encode_dialog_to_prompt();
        assert!(prompt.starts_with("<|system|>\nYou are helpful.<|end|>\n"));
        assert!(prompt.contains("<|user|>\nHi<|end|>\n"));
        assert!(prompt.ends_with("<|assistant|>\n"));
    }

    #[test]
    fn test_phi4_multi_turn() {
        let mut h = Phi4History::new();
        h.push(Message::user("Hello".into()));
        h.push(Message::assistant("Hi there!".into()));
        h.push(Message::user("Bye".into()));

        let prompt = h.encode_dialog_to_prompt();
        assert!(prompt.contains("<|assistant|>\nHi there!<|end|>\n"));
        assert!(prompt.contains("<|user|>\nBye<|end|>\n"));
        assert!(prompt.ends_with("<|assistant|>\n"));
    }

    #[test]
    fn test_phi4_clear() {
        let mut h = Phi4History::new();
        h.push(Message::user("test".into()));
        h.clear();
        let prompt = h.encode_dialog_to_prompt();
        // Empty history still primes assistant
        assert_eq!(prompt, "<|assistant|>\n");
    }
}
