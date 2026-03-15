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
    fn test_message_constructors() {
        let sys = Message::system("sys".into());
        assert!(matches!(sys.role, MessageRole::System));
        assert_eq!(sys.content, "sys");

        let usr = Message::user("usr".into());
        assert!(matches!(usr.role, MessageRole::User));

        let asst = Message::assistant("asst".into());
        assert!(matches!(asst.role, MessageRole::Assistant));
    }

    #[test]
    fn test_role_display() {
        assert_eq!(format!("{}", MessageRole::System), "system");
        assert_eq!(format!("{}", MessageRole::User), "user");
        assert_eq!(format!("{}", MessageRole::Assistant), "assistant");
    }

    #[test]
    fn test_message_json_roundtrip() {
        let msg = Message::user("Hello world".into());
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"content\":\"Hello world\""));

        let decoded: Message = serde_json::from_str(&json).unwrap();
        assert!(matches!(decoded.role, MessageRole::User));
        assert_eq!(decoded.content, "Hello world");
    }

    #[test]
    fn test_role_deserialize_lowercase() {
        let json = r#"{"role":"system","content":"test"}"#;
        let msg: Message = serde_json::from_str(json).unwrap();
        assert!(matches!(msg.role, MessageRole::System));
    }
}
