use std::env;

#[derive(Debug, Clone)]
pub struct Config {
    pub database_url: String,
    pub ai_service_url: String,
    pub upload_dir: String,
    pub export_dir: String,
    pub upload_max_body_size_mb: usize,
    pub host: String,
    pub port: u16,
    pub cors_origins: Vec<String>,
}

impl Config {
    pub fn from_env() -> Self {
        let cors_origins = env::var("CORS_ORIGINS")
            .unwrap_or_else(|_| "http://localhost:5173,http://localhost:80".into())
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        Self {
            database_url: env::var("DATABASE_URL")
                .unwrap_or_else(|_| "sqlite:storage/medassistant.db".into()),
            ai_service_url: env::var("AI_SERVICE_URL")
                .unwrap_or_else(|_| "http://localhost:8001".into()),
            upload_dir: env::var("UPLOAD_DIR").unwrap_or_else(|_| "storage/uploads".into()),
            export_dir: env::var("EXPORT_DIR").unwrap_or_else(|_| "storage/exports".into()),
            upload_max_body_size_mb: env::var("UPLOAD_MAX_BODY_SIZE_MB")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(50),
            host: env::var("HOST").unwrap_or_else(|_| "0.0.0.0".into()),
            port: env::var("PORT")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or(3000),
            cors_origins,
        }
    }

    pub fn upload_max_body_size_bytes(&self) -> usize {
        self.upload_max_body_size_mb.saturating_mul(1024 * 1024)
    }
}
