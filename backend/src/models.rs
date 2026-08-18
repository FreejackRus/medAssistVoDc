use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};

/// Full document with all fields (internal use, not sent to frontend)
#[allow(dead_code)]
#[derive(Debug, Serialize, Deserialize, sqlx::FromRow)]
pub struct Document {
    pub id: String,
    pub user_id: String,
    pub filename: String,
    pub file_path: String,
    pub diagnosis_name: Option<String>,
    pub mkb_code: Option<String>,
    pub full_text: Option<String>,
    pub status: String,
    pub chunk_count: i64,
    pub created_at: NaiveDateTime,
    pub updated_at: NaiveDateTime,
}

/// Document summary without full_text (sent to frontend)
#[derive(Debug, Serialize, Deserialize, sqlx::FromRow)]
pub struct DocumentSummary {
    pub id: String,
    pub user_id: String,
    pub filename: String,
    pub diagnosis_name: Option<String>,
    pub mkb_code: Option<String>,
    pub status: String,
    pub chunk_count: i64,
    pub created_at: NaiveDateTime,
    pub updated_at: NaiveDateTime,
}

#[derive(Debug, Serialize, Deserialize, sqlx::FromRow)]
pub struct ChatSession {
    pub id: String,
    pub user_id: String,
    pub document_id: String,
    pub title: String,
    pub created_at: NaiveDateTime,
}

#[derive(Debug, Serialize, Deserialize, sqlx::FromRow)]
pub struct ChatMessage {
    pub id: String,
    pub session_id: String,
    pub role: String,
    pub content: String,
    pub status: String,
    pub stream_seq: i64,
    pub created_at: NaiveDateTime,
}

#[derive(Debug, Serialize, Deserialize, sqlx::FromRow)]
pub struct Algorithm {
    pub id: String,
    pub document_id: String,
    pub content_markdown: String,
    pub status: String,
    pub generation_mode: String,
    pub stream_seq: i64,
    pub created_at: NaiveDateTime,
}

#[derive(Debug, Serialize, Deserialize, sqlx::FromRow)]
pub struct AdminUserSummary {
    pub id: String,
    pub username: String,
    pub role: String,
    pub must_change_password: bool,
    pub created_by: Option<String>,
    pub creator_username: Option<String>,
    pub display_name: Option<String>,
    pub organization: Option<String>,
    pub position: Option<String>,
    pub notes: Option<String>,
    pub profile_fields: String,
    pub allowed_profile_fields: String,
    pub onboarding_expires_at: Option<NaiveDateTime>,
    pub created_at: NaiveDateTime,
    pub updated_at: NaiveDateTime,
    pub last_login_at: Option<NaiveDateTime>,
    pub active_sessions: i64,
    pub documents_count: Option<i64>,
    pub algorithm_generations_count: Option<i64>,
    pub chat_generations_count: Option<i64>,
    pub documents_with_algorithm_count: Option<i64>,
    pub documents_with_chat_count: Option<i64>,
}
