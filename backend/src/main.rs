use std::net::SocketAddr;

use axum::http::HeaderValue;
use axum::{
    Router,
    extract::DefaultBodyLimit,
    routing::{get, post},
};
use sqlx::SqlitePool;
use tower_http::{
    cors::{Any, CorsLayer},
    trace::TraceLayer,
};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod auth;
mod config;
mod db;
mod error;
mod models;
mod python_client;
mod routes;
mod sse;

use config::Config;
use python_client::PythonClient;

#[derive(Clone)]
pub struct AppState {
    pub db: SqlitePool,
    pub python: PythonClient,
    pub config: Config,
}

#[tokio::main]
async fn main() {
    dotenvy::dotenv().ok();

    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let config = Config::from_env();

    // Ensure storage directories exist
    std::fs::create_dir_all(&config.upload_dir).ok();
    std::fs::create_dir_all(&config.export_dir).ok();

    let pool = db::init_pool(&config.database_url).await;
    db::run_migrations(&pool).await;
    auth::ensure_bootstrap_admin(&pool).await;

    let python = PythonClient::new(&config.ai_service_url);

    let state = AppState {
        db: pool,
        python,
        config: config.clone(),
    };

    let origins: Vec<HeaderValue> = config
        .cors_origins
        .iter()
        .filter_map(|o| o.parse().ok())
        .collect();

    let cors = CorsLayer::new()
        .allow_origin(origins)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/health", get(routes::health::health_check))
        .route("/api/config", get(routes::app_config::get_config))
        // Auth
        .route("/api/auth/login", post(routes::auth::login))
        .route("/api/auth/logout", post(routes::auth::logout))
        .route("/api/auth/me", get(routes::auth::me))
        .route(
            "/api/auth/profile",
            axum::routing::patch(routes::auth::update_profile),
        )
        .route(
            "/api/auth/change-password",
            post(routes::auth::change_password),
        )
        .route(
            "/api/auth/complete-onboarding",
            post(routes::auth::complete_onboarding),
        )
        .route("/api/events", get(routes::events::stream))
        // Admin
        .route(
            "/api/admin/users",
            get(routes::admin::list_users).post(routes::admin::create_user),
        )
        .route(
            "/api/admin/users/{user_id}",
            axum::routing::patch(routes::admin::update_user).delete(routes::admin::delete_user),
        )
        .route(
            "/api/admin/users/{user_id}/reset-password",
            post(routes::admin::reset_password),
        )
        .route("/api/monitoring", get(routes::monitoring::summary))
        // Documents
        .route(
            "/api/documents/upload",
            post(routes::documents::upload)
                .layer(DefaultBodyLimit::max(config.upload_max_body_size_bytes())),
        )
        .route(
            "/api/documents/import-recommendation",
            post(routes::documents::import_recommendation),
        )
        .route("/api/documents", get(routes::documents::list))
        .route(
            "/api/documents/{doc_id}",
            get(routes::documents::get).delete(routes::documents::delete),
        )
        .route(
            "/api/documents/{doc_id}/retry",
            post(routes::documents::retry),
        )
        // Algorithms
        .route(
            "/api/algorithms/generate",
            post(routes::algorithms::generate),
        )
        .route(
            "/api/algorithms/by-document/{doc_id}",
            get(routes::algorithms::get_by_document),
        )
        .route(
            "/api/algorithms/{algo_id}/stream",
            get(routes::algorithms::stream),
        )
        .route("/api/algorithms/{algo_id}", get(routes::algorithms::get))
        .route(
            "/api/algorithms/export-pdf",
            post(routes::algorithms::export_pdf),
        )
        // Chat
        .route(
            "/api/chat/sessions",
            post(routes::chat::create_session).get(routes::chat::list_sessions),
        )
        .route(
            "/api/chat/sessions/{session_id}/messages",
            get(routes::chat::get_messages)
                .post(routes::chat::send_message)
                .layer(DefaultBodyLimit::max(config.upload_max_body_size_bytes())),
        )
        .route(
            "/api/chat/messages/{message_id}/stream",
            get(routes::chat::stream_message),
        )
        .route(
            "/api/chat/sessions/{session_id}",
            axum::routing::delete(routes::chat::delete_session),
        )
        // Calculators
        .route("/api/calculators/bmi", post(routes::calculators::calc_bmi))
        .route(
            "/api/calculators/creatinine",
            post(routes::calculators::calc_creatinine),
        )
        .route("/api/calculators/bsa", post(routes::calculators::calc_bsa))
        .route(
            "/api/calculators/dosage",
            post(routes::calculators::calc_dosage),
        )
        .route(
            "/api/calculators/score",
            post(routes::calculators::calc_score),
        )
        // Services
        .route(
            "/api/services/match",
            post(routes::services::match_services),
        )
        // Clinical Recommendations
        .route(
            "/api/clinical-recommendations",
            get(routes::clinical_recs::get_recommendations),
        )
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr: SocketAddr = format!("{}:{}", config.host, config.port)
        .parse()
        .expect("Invalid address");

    tracing::info!("Backend listening on {addr}");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
