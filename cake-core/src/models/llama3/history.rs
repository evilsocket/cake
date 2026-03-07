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
    fn test_empty_history_has_assistant_header() {
        let history = History::new();
        let prompt = history.encode_dialog_to_prompt();
        assert!(prompt.starts_with("<|begin_of_text|>"));
        assert!(prompt.contains("<|start_header_id|>assistant<|end_header_id|>"));
    }

    #[test]
    fn test_single_turn_encoding() {
        let mut history = History::new();
        history.push(Message::system("You are helpful.".into()));
        history.push(Message::user("Hello".into()));

        let prompt = history.encode_dialog_to_prompt();

        assert!(prompt.starts_with("<|begin_of_text|>"));
        assert!(prompt.contains("<|start_header_id|>system<|end_header_id|>\n\nYou are helpful.<|eot_id|>"));
        assert!(prompt.contains("<|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|>"));
        assert!(prompt.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn test_multi_turn_encoding() {
        let mut history = History::new();
        history.push(Message::system("Sys".into()));
        history.push(Message::user("Q1".into()));
        history.push(Message::assistant("A1".into()));
        history.push(Message::user("Q2".into()));

        let prompt = history.encode_dialog_to_prompt();

        // All messages present in order
        let sys_pos = prompt.find("Sys").unwrap();
        let q1_pos = prompt.find("Q1").unwrap();
        let a1_pos = prompt.find("A1").unwrap();
        let q2_pos = prompt.find("Q2").unwrap();
        assert!(sys_pos < q1_pos);
        assert!(q1_pos < a1_pos);
        assert!(a1_pos < q2_pos);

        // Ends with assistant header for completion
        assert!(prompt.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn test_whitespace_trimmed() {
        let mut history = History::new();
        history.push(Message::user("  hello  ".into()));

        let prompt = history.encode_dialog_to_prompt();
        assert!(prompt.contains("hello<|eot_id|>"));
        // Leading/trailing whitespace in content should be trimmed
        assert!(!prompt.contains("  hello"));
    }
}
