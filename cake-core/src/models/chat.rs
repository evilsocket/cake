use serde::{Deserialize, Serialize};

/// The role of a message in a chat.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    /// System prompt.
    System,
    /// User prompt.
    User,
    /// Assistant response.
    Assistant,
}

impl std::fmt::Display for MessageRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                MessageRole::System => "system",
                MessageRole::User => "user",
                MessageRole::Assistant => "assistant",
            }
        )
    }
}

/// A chat message.
#[derive(Debug, Serialize, Deserialize)]
pub struct Message {
    /// Message role.
    pub role: MessageRole,
    /// Messagae content.
    pub content: String,
}

impl Message {
    /// Create a system message.
    pub fn system(content: String) -> Self {
        Self {
            role: MessageRole::System,
            content,
        }
    }

    /// Create a user message.
    pub fn user(content: String) -> Self {
        Self {
            role: MessageRole::User,
            content,
        }
    }

    /// Create an assistant message.
    pub fn assistant(content: String) -> Self {
        Self {
            role: MessageRole::Assistant,
            content,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_role_display() {
        assert_eq!(format!("{}", MessageRole::System), "system");
        assert_eq!(format!("{}", MessageRole::User), "user");
        assert_eq!(format!("{}", MessageRole::Assistant), "assistant");
    }

    #[test]
    fn test_message_constructors() {
        let sys = Message::system("prompt".into());
        assert_eq!(sys.role, MessageRole::System);
        assert_eq!(sys.content, "prompt");

        let usr = Message::user("hello".into());
        assert_eq!(usr.role, MessageRole::User);
        assert_eq!(usr.content, "hello");

        let asst = Message::assistant("hi".into());
        assert_eq!(asst.role, MessageRole::Assistant);
        assert_eq!(asst.content, "hi");
    }

    #[test]
    fn test_message_role_serde() {
        let json = r#"{"role":"system","content":"test"}"#;
        let msg: Message = serde_json::from_str(json).unwrap();
        assert_eq!(msg.role, MessageRole::System);
        assert_eq!(msg.content, "test");

        let json = r#"{"role":"user","content":"hello"}"#;
        let msg: Message = serde_json::from_str(json).unwrap();
        assert_eq!(msg.role, MessageRole::User);

        let json = r#"{"role":"assistant","content":"hi"}"#;
        let msg: Message = serde_json::from_str(json).unwrap();
        assert_eq!(msg.role, MessageRole::Assistant);
    }

    #[test]
    fn test_message_serialize_roundtrip() {
        let msg = Message::user("hello world".into());
        let json = serde_json::to_string(&msg).unwrap();
        let msg2: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(msg2.role, MessageRole::User);
        assert_eq!(msg2.content, "hello world");
    }
}
