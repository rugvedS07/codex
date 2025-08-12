use codex_core::LMSTUDIO_PROVIDER_ID;
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
            .get(LMSTUDIO_PROVIDER_ID)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("Built-in provider {LMSTUDIO_PROVIDER_ID} not found",),
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
}
