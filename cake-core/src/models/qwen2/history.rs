use crate::models::chat::Message;

/// ChatML chat history for Qwen2 models.
pub struct QwenHistory(Vec<Message>);

// ChatML format used by Qwen2:
// <|im_start|>system
// You are a helpful assistant.<|im_end|>
// <|im_start|>user
// Hello<|im_end|>
// <|im_start|>assistant
impl QwenHistory {
    fn encode_message(message: &Message) -> String {
        format!(
            "<|im_start|>{}\n{}<|im_end|>",
            message.role,
            message.content.trim()
        )
    }

    /// Create a new instance of this object.
    pub fn new() -> Self {
        Self(vec![])
    }

    /// Encode the dialog to ChatML prompt format.
    pub fn encode_dialog_to_prompt(&self) -> String {
        let mut encoded = String::new();

        for message in self.iter() {
            encoded += &Self::encode_message(message);
            encoded += "\n";
        }

        //  Add the start of an assistant message for the model to complete.
        encoded += "<|im_start|>assistant\n";

        encoded
    }
}

impl std::ops::Deref for QwenHistory {
    type Target = Vec<Message>;
    fn deref(&self) -> &Vec<Message> {
        &self.0
    }
}

impl std::ops::DerefMut for QwenHistory {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chatml_encoding() {
        let mut history = QwenHistory::new();
        history.push(Message::system("You are a helpful assistant.".into()));
        history.push(Message::user("Hello".into()));

        let prompt = history.encode_dialog_to_prompt();
        assert!(prompt.contains("<|im_start|>system\nYou are a helpful assistant.<|im_end|>"));
        assert!(prompt.contains("<|im_start|>user\nHello<|im_end|>"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }
}
