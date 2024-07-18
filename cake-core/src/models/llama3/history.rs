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
