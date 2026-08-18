use std::time::Duration;

use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Clone)]
pub struct PythonClient {
    /// Client with timeout — for regular requests (ingest, recs, services, health, delete)
    client: Client,
    /// Client without overall timeout — for SSE streams (algorithm generation, chat)
    stream_client: Client,
    base_url: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub ollama: String,
}

impl PythonClient {
    pub fn new(base_url: &str) -> Self {
        let client = Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(300))
            .build()
            .expect("Failed to build HTTP client");

        let stream_client = Client::builder()
            .connect_timeout(Duration::from_secs(10))
            // No overall timeout — SSE streams can run for minutes on CPU
            .build()
            .expect("Failed to build stream HTTP client");

        Self {
            client,
            stream_client,
            base_url: base_url.trim_end_matches('/').to_string(),
        }
    }

    pub fn url(&self, path: &str) -> String {
        format!("{}/api/v1{}", self.base_url, path)
    }

    /// Client with 5-minute timeout for regular requests
    pub fn client(&self) -> &Client {
        &self.client
    }

    /// Client without timeout for SSE streaming (algorithm, chat)
    pub fn stream_client(&self) -> &Client {
        &self.stream_client
    }

    pub async fn health(&self) -> Result<HealthResponse, reqwest::Error> {
        self.client
            .get(self.url("/health"))
            .timeout(Duration::from_secs(5))
            .send()
            .await?
            .json()
            .await
    }
}
