use serde::{Deserialize, Serialize};

/// The role of a message in a chat.
#[derive(Debug, Serialize, Deserialize)]
pub enum MessageRole {
    /// System prompt.
    #[serde(alias = "system")]
    System,
    /// User prompt.
    #[serde(alias = "user")]
    User,
    /// Assistant response.
    #[serde(alias = "assistant")]
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
