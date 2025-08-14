use codex_core::LMSTUDIO_OSS_PROVIDER_ID;
use codex_core::config::Config;
use std::io;

pub struct LMStudioClient {
    client: reqwest::Client,
    base_url: String,
}

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
        let base_url = provider
            .base_url
            .as_ref()
            .expect("oss provider must have a base_url");

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

    async fn check_server(&self) -> io::Result<()> {
        let url = format!("{}/models", self.base_url.trim_end_matches('/'));
        let response = self.client.get(&url).send().await;

        match response {
            Ok(resp) if resp.status().is_success() => Ok(()),
            Ok(resp) => Err(io::Error::new(
                io::ErrorKind::Other,
                format!("Server returned error: {}", resp.status()),
            )),
            Err(err) => Err(io::Error::new(io::ErrorKind::Other, err)),
        }
    }

    // Return the list of models available on the LM Studio server.
    pub async fn fetch_models(&self) -> io::Result<Vec<String>> {
        let url = format!("{}/models", self.base_url.trim_end_matches('/'));
        let response =
            self.client.get(&url).send().await.map_err(|e| {
                io::Error::new(io::ErrorKind::Other, format!("Request failed: {e}"))
            })?;

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
                .map(|id| id.to_string())
                .collect();
            Ok(models)
        } else {
            Err(io::Error::new(
                io::ErrorKind::Other,
                format!("Failed to fetch models: {}", response.status()),
            ))
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
            .and(wiremock::matchers::path("/models"))
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
            .and(wiremock::matchers::path("/models"))
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
            .and(wiremock::matchers::path("/models"))
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
            .and(wiremock::matchers::path("/models"))
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
            .and(wiremock::matchers::path("/models"))
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
}
