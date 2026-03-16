use crate::models::chat::Message;

/// Chat history.
pub struct History(Vec<Message>);

// Adapted from https://github.com/meta-llama/llama3/blob/main/llama/tokenizer.py#L202
impl History {
    fn encode_header(message: &Message) -> String {
        format!("<|start_header_id|>{}<|end_header_id|>\n\n", message.role)
    }

    fn encode_message(message: &Message) -> String {
        Self::encode_header(message) + message.content.trim() + "<|eot_id|>"
    }

    /// Create a new instance of this object.
    pub fn new() -> Self {
        Self(vec![])
    }

    /// Encode the dialog to llama3 prompt format.
    pub fn encode_dialog_to_prompt(&self) -> String {
        let mut encoded = "<|begin_of_text|>".to_string();

        for message in self.iter() {
            encoded += &Self::encode_message(message);
        }

        //  Add the start of an assistant message for the model to complete.
        encoded += &Self::encode_header(&Message::assistant("".to_string()));

        encoded
    }
}

impl std::ops::Deref for History {
    type Target = Vec<Message>;
    fn deref(&self) -> &Vec<Message> {
        &self.0
    }
}

impl std::ops::DerefMut for History {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_header() {
        let msg = Message::user("hello".into());
        let header = History::encode_header(&msg);
        assert_eq!(header, "<|start_header_id|>user<|end_header_id|>\n\n");
    }

    #[test]
    fn test_encode_message() {
        let msg = Message::system("You are helpful.".into());
        let encoded = History::encode_message(&msg);
        assert_eq!(
            encoded,
            "<|start_header_id|>system<|end_header_id|>\n\nYou are helpful.<|eot_id|>"
        );
    }

    #[test]
    fn test_encode_dialog_to_prompt() {
        let mut history = History::new();
        history.push(Message::system("Be brief.".into()));
        history.push(Message::user("Hi".into()));

        let prompt = history.encode_dialog_to_prompt();
        assert!(prompt.starts_with("<|begin_of_text|>"));
        assert!(prompt.contains("<|start_header_id|>system<|end_header_id|>\n\nBe brief.<|eot_id|>"));
        assert!(prompt.contains("<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>"));
        // Ends with assistant header to prime generation
        assert!(prompt.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn test_empty_dialog() {
        let history = History::new();
        let prompt = history.encode_dialog_to_prompt();
        assert_eq!(
            prompt,
            "<|begin_of_text|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }
}
