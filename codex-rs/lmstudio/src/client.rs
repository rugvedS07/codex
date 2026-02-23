use codex_core::LMSTUDIO_OSS_PROVIDER_ID;
use codex_core::config::Config;
use std::io;
use std::io::Write;

#[derive(Clone)]
pub struct LMStudioClient {
    client: reqwest::Client,
    base_url: String,
}

const LMSTUDIO_CONNECTION_ERROR: &str = "LM Studio is not responding. Install from https://lmstudio.ai/download and run 'lms server start'.";
// 8192 tokens provides headroom above the observed ~6000 token initial input.
const DEFAULT_CONTEXT_LENGTH: u32 = 8192;

impl LMStudioClient {
    pub async fn try_from_provider(config: &Config) -> std::io::Result<Self> {
        let provider = config
            .model_providers
            .get(LMSTUDIO_OSS_PROVIDER_ID)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("Built-in provider {LMSTUDIO_OSS_PROVIDER_ID} not found",),
                )
            })?;
        let base_url = provider.base_url.as_ref().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "oss provider must have a base_url",
            )
        })?;

        let client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(5))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        let client = LMStudioClient {
            client,
            base_url: base_url.to_string(),
        };
        client.check_server().await?;

        Ok(client)
    }

    fn host_root(&self) -> String {
        let base_url = self.base_url.trim_end_matches('/');
        base_url.strip_suffix("/v1").unwrap_or(base_url).to_string()
    }

    async fn check_server(&self) -> io::Result<()> {
        let url = format!("{}/v1/models", self.host_root());
        let response = self.client.get(&url).send().await;

        if let Ok(resp) = response {
            if resp.status().is_success() {
                Ok(())
            } else {
                Err(io::Error::other(format!(
                    "Server returned error: {} {LMSTUDIO_CONNECTION_ERROR}",
                    resp.status()
                )))
            }
        } else {
            Err(io::Error::other(LMSTUDIO_CONNECTION_ERROR))
        }
    }

    // Check if a model is already loaded with the same key
    async fn is_model_loaded(&self, model: &str) -> bool {
        let models_url = format!("{}/api/v1/models", self.host_root());
        let Ok(response) = self.client.get(&models_url).send().await else {
            return false;
        };
        if !response.status().is_success() {
            return false;
        }
        let Ok(json) = response.json::<serde_json::Value>().await else {
            return false;
        };
        let models = json.get("models").and_then(|value| value.as_array());
        models.is_some_and(|entries| {
            entries.iter().any(|entry| {
                let loaded_instances = entry
                    .get("loaded_instances")
                    .and_then(|value| value.as_array());
                // A model is considered loaded if any of its loaded_instances
                // shares the same id as the requested model.
                loaded_instances.is_some_and(|instances| {
                    instances.iter().any(|instance| {
                        instance.get("id").and_then(|value| value.as_str()) == Some(model)
                    })
                })
            })
        })
    }

    // Load a model by sending an empty request with max_tokens 1
    pub async fn load_model(&self, model: &str) -> io::Result<()> {
        if self.is_model_loaded(model).await {
            tracing::info!("Model '{model}' already loaded; reusing existing instance");
            return Ok(());
        }
        let url = format!("{}/api/v1/models/load", self.host_root());

        let request_body = serde_json::json!({
            "model": model,
            "context_length": DEFAULT_CONTEXT_LENGTH
        });

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| io::Error::other(format!("Request failed: {e}")))?;

        if response.status().is_success() {
            tracing::info!("Successfully loaded model '{model}'");
            Ok(())
        } else {
            Err(io::Error::other(format!(
                "Failed to load model: {}",
                response.status()
            )))
        }
    }

    // Return the list of models available on the LM Studio server.
    pub async fn fetch_models(&self) -> io::Result<Vec<String>> {
        let url = format!("{}/v1/models", self.host_root());
        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| io::Error::other(format!("Request failed: {e}")))?;

        if response.status().is_success() {
            let json: serde_json::Value = response.json().await.map_err(|e| {
                io::Error::new(io::ErrorKind::InvalidData, format!("JSON parse error: {e}"))
            })?;
            let models = json["data"]
                .as_array()
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "No 'data' array in response")
                })?
                .iter()
                .filter_map(|model| model["id"].as_str())
                .map(std::string::ToString::to_string)
                .collect();
            Ok(models)
        } else {
            Err(io::Error::other(format!(
                "Failed to fetch models: {}",
                response.status()
            )))
        }
    }

    pub async fn download_model(&self, model: &str) -> std::io::Result<()> {
        let url = format!("{}/api/v1/models/download", self.host_root());

        let request_body = serde_json::json!({
            "model": model
        });

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| io::Error::other(format!("Request failed: {e}")))?;

        if !response.status().is_success() {
            return Err(io::Error::other(format!(
                "Failed to download model: {}",
                response.status()
            )));
        }

        let download_status = response.json::<serde_json::Value>().await.map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("JSON parse error: {e}"))
        })?;

        type DownloadStatus = (String, Option<String>, Option<u64>, Option<u64>);

        let parse_status = |json: &serde_json::Value| -> io::Result<DownloadStatus> {
            let status = json["status"]
                .as_str()
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing status"))?
                .to_string();
            let job_id = json["job_id"]
                .as_str()
                .map(std::string::ToString::to_string);
            let downloaded_bytes = json["downloaded_bytes"].as_u64();
            let total_size_bytes = json["total_size_bytes"].as_u64();
            Ok((status, job_id, downloaded_bytes, total_size_bytes))
        };

        let (status, job_id, _, _) = parse_status(&download_status)?;

        match status.as_str() {
            "already_downloaded" | "completed" => {
                tracing::info!("Model '{model}' is ready");
                Ok(())
            }
            "failed" => Err(io::Error::other(format!(
                "Model download failed for '{model}'"
            ))),
            "paused" => Err(io::Error::other(format!(
                "Model download paused for '{model}'"
            ))),
            "downloading" => {
                let job_id = job_id.as_deref().ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "Download status missing job_id")
                })?;

                let mut last_logged =
                    std::time::Instant::now() - std::time::Duration::from_secs(10);

                loop {
                    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                    let status_url = format!(
                        "{}/api/v1/models/download/status/{job_id}",
                        self.host_root()
                    );

                    let status_response = self
                        .client
                        .get(&status_url)
                        .send()
                        .await
                        .map_err(|e| io::Error::other(format!("Request failed: {e}")))?;

                    if !status_response.status().is_success() {
                        return Err(io::Error::other(format!(
                            "Failed to fetch download status: {}",
                            status_response.status()
                        )));
                    }

                    let status =
                        status_response
                            .json::<serde_json::Value>()
                            .await
                            .map_err(|e| {
                                io::Error::new(
                                    io::ErrorKind::InvalidData,
                                    format!("JSON parse error: {e}"),
                                )
                            })?;
                    let (status_value, _, downloaded_bytes, total_size_bytes) =
                        parse_status(&status)?;

                    match status_value.as_str() {
                        "completed" => {
                            eprintln!();
                            tracing::info!("Successfully downloaded model '{model}'");
                            return Ok(());
                        }
                        "failed" => {
                            eprintln!();
                            return Err(io::Error::other(format!(
                                "Model download failed for '{model}'"
                            )));
                        }
                        "paused" => {
                            eprintln!();
                            return Err(io::Error::other(format!(
                                "Model download paused for '{model}'"
                            )));
                        }
                        "downloading" => {
                            if let (Some(downloaded), Some(total)) =
                                (downloaded_bytes, total_size_bytes)
                            {
                                let now = std::time::Instant::now();
                                if now.duration_since(last_logged)
                                    >= std::time::Duration::from_millis(500)
                                {
                                    let percent = (downloaded as f64 / total as f64) * 100.0;
                                    eprint!(
                                        "\rDownloading '{model}': {} / {} ({percent:.1}%)",
                                        format_bytes(downloaded),
                                        format_bytes(total)
                                    );
                                    let _ = std::io::stderr().flush();
                                    last_logged = now;
                                }
                            }
                        }
                        status_value => {
                            eprintln!();
                            return Err(io::Error::other(format!(
                                "Unknown download status '{status_value}' for '{model}'"
                            )));
                        }
                    }
                }
            }
            status_value => Err(io::Error::other(format!(
                "Unknown download status '{status_value}' for '{model}'"
            ))),
        }
    }

    /// Low-level constructor given a raw host root, e.g. "http://localhost:1234".
    #[cfg(test)]
    fn from_host_root(host_root: impl Into<String>) -> Self {
        let client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(5))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self {
            client,
            base_url: host_root.into(),
        }
    }
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KiB", "MiB", "GiB", "TiB"];
    let mut size = bytes as f64;
    let mut i = 0;
    while size >= 1024.0 && i < UNITS.len() - 1 {
        size /= 1024.0;
        i += 1;
    }
    if i == 0 {
        format!("{size} B")
    } else {
        format!("{size:.2} {}", UNITS[i])
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::unwrap_used)]
    use super::*;

    #[tokio::test]
    async fn test_fetch_models_happy_path() {
        if std::env::var(codex_core::spawn::CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR).is_ok() {
            tracing::info!(
                "{} is set; skipping test_fetch_models_happy_path",
                codex_core::spawn::CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR
            );
            return;
        }

        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .and(wiremock::matchers::path("/v1/models"))
            .respond_with(
                wiremock::ResponseTemplate::new(200).set_body_raw(
                    serde_json::json!({
                        "data": [
                            {"id": "openai/gpt-oss-20b"},
                        ]
                    })
                    .to_string(),
                    "application/json",
                ),
            )
            .mount(&server)
            .await;

        let client = LMStudioClient::from_host_root(server.uri());
        let models = client.fetch_models().await.expect("fetch models");
        assert!(models.contains(&"openai/gpt-oss-20b".to_string()));
    }

    #[tokio::test]
    async fn test_fetch_models_no_data_array() {
        if std::env::var(codex_core::spawn::CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR).is_ok() {
            tracing::info!(
                "{} is set; skipping test_fetch_models_no_data_array",
                codex_core::spawn::CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR
            );
            return;
        }

        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .and(wiremock::matchers::path("/v1/models"))
            .respond_with(
                wiremock::ResponseTemplate::new(200)
                    .set_body_raw(serde_json::json!({}).to_string(), "application/json"),
            )
            .mount(&server)
            .await;

        let client = LMStudioClient::from_host_root(server.uri());
        let result = client.fetch_models().await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("No 'data' array in response")
        );
    }

    #[tokio::test]
    async fn test_fetch_models_server_error() {
        if std::env::var(codex_core::spawn::CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR).is_ok() {
            tracing::info!(
                "{} is set; skipping test_fetch_models_server_error",
                codex_core::spawn::CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR
            );
            return;
        }

        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .and(wiremock::matchers::path("/v1/models"))
            .respond_with(wiremock::ResponseTemplate::new(500))
            .mount(&server)
            .await;

        let client = LMStudioClient::from_host_root(server.uri());
        let result = client.fetch_models().await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Failed to fetch models: 500")
        );
    }

    #[tokio::test]
    async fn test_check_server_happy_path() {
        if std::env::var(codex_core::spawn::CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR).is_ok() {
            tracing::info!(
                "{} is set; skipping test_check_server_happy_path",
                codex_core::spawn::CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR
            );
            return;
        }

        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .and(wiremock::matchers::path("/v1/models"))
            .respond_with(wiremock::ResponseTemplate::new(200))
            .mount(&server)
            .await;

        let client = LMStudioClient::from_host_root(server.uri());
        client
            .check_server()
            .await
            .expect("server check should pass");
    }

    #[tokio::test]
    async fn test_check_server_error() {
        if std::env::var(codex_core::spawn::CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR).is_ok() {
            tracing::info!(
                "{} is set; skipping test_check_server_error",
                codex_core::spawn::CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR
            );
            return;
        }

        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .and(wiremock::matchers::path("/v1/models"))
            .respond_with(wiremock::ResponseTemplate::new(404))
            .mount(&server)
            .await;

        let client = LMStudioClient::from_host_root(server.uri());
        let result = client.check_server().await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Server returned error: 404")
        );
    }

    #[tokio::test]
    async fn test_load_model_happy_path() {
        if std::env::var(codex_core::spawn::CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR).is_ok() {
            tracing::info!(
                "{} is set; skipping test_load_model_happy_path",
                codex_core::spawn::CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR
            );
            return;
        }

        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/api/v1/models/load"))
            .respond_with(wiremock::ResponseTemplate::new(200))
            .mount(&server)
            .await;

        let client = LMStudioClient::from_host_root(format!("{uri}/v1", uri = server.uri()));
        client
            .load_model("openai/gpt-oss-20b")
            .await
            .expect("load model");
    }

    #[tokio::test]
    async fn test_load_model_reuses_loaded_instance() {
        if std::env::var(codex_core::spawn::CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR).is_ok() {
            tracing::info!(
                "{} is set; skipping test_load_model_reuses_loaded_instance",
                codex_core::spawn::CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR
            );
            return;
        }

        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .and(wiremock::matchers::path("/api/v1/models"))
            .respond_with(
                wiremock::ResponseTemplate::new(200).set_body_raw(
                    serde_json::json!({
                        "models": [
                            {
                                "key": "test/test-model",
                                "loaded_instances": [
                                    {
                                        "id": "test/test-model",
                                        "config": {
                                            "context_length": 7000
                                        }
                                    }
                                ]
                            }
                        ]
                    })
                    .to_string(),
                    "application/json",
                ),
            )
            .mount(&server)
            .await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/api/v1/models/load"))
            .respond_with(wiremock::ResponseTemplate::new(500))
            .mount(&server)
            .await;

        let client = LMStudioClient::from_host_root(server.uri());
        let result = client.load_model("test/test-model").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_is_model_loaded_true() {
        if std::env::var(codex_core::spawn::CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR).is_ok() {
            tracing::info!(
                "{} is set; skipping test_is_model_loaded_true",
                codex_core::spawn::CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR
            );
            return;
        }

        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .and(wiremock::matchers::path("/api/v1/models"))
            .respond_with(
                wiremock::ResponseTemplate::new(200).set_body_raw(
                    serde_json::json!({
                        "models": [
                            {
                                "key": "test/test-model",
                                "loaded_instances": [
                                    {
                                        "id": "test/test-model",
                                        "config": {
                                            "context_length": 7000
                                        }
                                    }
                                ]
                            }
                        ]
                    })
                    .to_string(),
                    "application/json",
                ),
            )
            .mount(&server)
            .await;

        let client = LMStudioClient::from_host_root(server.uri());
        assert!(client.is_model_loaded("test/test-model").await);
    }

    #[tokio::test]
    async fn test_is_model_loaded_false() {
        if std::env::var(codex_core::spawn::CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR).is_ok() {
            tracing::info!(
                "{} is set; skipping test_is_model_loaded_false",
                codex_core::spawn::CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR
            );
            return;
        }

        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .and(wiremock::matchers::path("/api/v1/models"))
            .respond_with(
                wiremock::ResponseTemplate::new(200).set_body_raw(
                    serde_json::json!({
                        "models": [
                            {
                                "key": "test/test-model",
                                "loaded_instances": []
                            }
                        ]
                    })
                    .to_string(),
                    "application/json",
                ),
            )
            .mount(&server)
            .await;

        let client = LMStudioClient::from_host_root(server.uri());
        assert!(!client.is_model_loaded("test/test-model").await);
    }

    #[tokio::test]
    async fn test_load_model_error() {
        if std::env::var(codex_core::spawn::CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR).is_ok() {
            tracing::info!(
                "{} is set; skipping test_load_model_error",
                codex_core::spawn::CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR
            );
            return;
        }

        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/api/v1/models/load"))
            .respond_with(wiremock::ResponseTemplate::new(500))
            .mount(&server)
            .await;

        let client = LMStudioClient::from_host_root(format!("{uri}/v1", uri = server.uri()));
        let result = client.load_model("openai/gpt-oss-20b").await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Failed to load model: 500")
        );
    }

    #[tokio::test]
    async fn test_download_model_happy_path() {
        if std::env::var(codex_core::spawn::CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR).is_ok() {
            tracing::info!(
                "{} is set; skipping test_download_model_happy_path",
                codex_core::spawn::CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR
            );
            return;
        }

        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/api/v1/models/download"))
            .respond_with(
                wiremock::ResponseTemplate::new(200).set_body_raw(
                    serde_json::json!({
                        "job_id": "job-1",
                        "status": "downloading"
                    })
                    .to_string(),
                    "application/json",
                ),
            )
            .mount(&server)
            .await;

        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .and(wiremock::matchers::path(
                "/api/v1/models/download/status/job-1",
            ))
            .respond_with(
                wiremock::ResponseTemplate::new(200).set_body_raw(
                    serde_json::json!({
                        "job_id": "job-1",
                        "status": "completed"
                    })
                    .to_string(),
                    "application/json",
                ),
            )
            .mount(&server)
            .await;

        let client = LMStudioClient::from_host_root(format!("{uri}/v1", uri = server.uri()));
        client
            .download_model("openai/gpt-oss-20b")
            .await
            .expect("download model");
    }

    #[tokio::test]
    async fn test_download_model_error() {
        if std::env::var(codex_core::spawn::CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR).is_ok() {
            tracing::info!(
                "{} is set; skipping test_download_model_error",
                codex_core::spawn::CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR
            );
            return;
        }

        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/api/v1/models/download"))
            .respond_with(
                wiremock::ResponseTemplate::new(200).set_body_raw(
                    serde_json::json!({
                        "job_id": "job-1",
                        "status": "downloading"
                    })
                    .to_string(),
                    "application/json",
                ),
            )
            .mount(&server)
            .await;

        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .and(wiremock::matchers::path(
                "/api/v1/models/download/status/job-1",
            ))
            .respond_with(
                wiremock::ResponseTemplate::new(200).set_body_raw(
                    serde_json::json!({
                        "job_id": "job-1",
                        "status": "failed"
                    })
                    .to_string(),
                    "application/json",
                ),
            )
            .mount(&server)
            .await;

        let client = LMStudioClient::from_host_root(format!("{uri}/v1", uri = server.uri()));
        let result = client.download_model("openai/gpt-oss-20b").await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Model download failed")
        );
    }

    #[test]
    fn test_from_host_root() {
        let client = LMStudioClient::from_host_root("http://localhost:1234");
        assert_eq!(client.base_url, "http://localhost:1234");

        let client = LMStudioClient::from_host_root("https://example.com:8080/api");
        assert_eq!(client.base_url, "https://example.com:8080/api");
    }
}
