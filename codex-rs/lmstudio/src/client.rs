use codex_core::LMSTUDIO_OSS_PROVIDER_ID;
use codex_core::config::Config;
use codex_core::models_manager::model_info::BASE_INSTRUCTIONS;
use codex_protocol::config_types::ReasoningSummary;
use codex_protocol::openai_models::ApplyPatchToolType;
use codex_protocol::openai_models::ConfigShellToolType;
use codex_protocol::openai_models::InputModality;
use codex_protocol::openai_models::ModelInfo;
use codex_protocol::openai_models::ModelVisibility;
use codex_protocol::openai_models::ReasoningEffort;
use codex_protocol::openai_models::ReasoningEffortPreset;
use codex_protocol::openai_models::TruncationPolicyConfig;
use serde::Deserialize;
use std::io;
use std::io::Write;
use std::time::Duration;
use std::time::Instant;

#[derive(Clone)]
pub struct LMStudioClient {
    client: reqwest::Client,
    base_url: String,
}

const LMSTUDIO_CONNECTION_ERROR: &str = "LM Studio is not responding. Install from https://lmstudio.ai/download and run 'lms server start'.";
const DEFAULT_CONTEXT_LENGTH: i64 = 64000;

impl LMStudioClient {
    pub async fn try_from_provider(config: &Config) -> io::Result<Self> {
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
            .connect_timeout(Duration::from_secs(5))
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
        let url = format!("{}/api/v1/models", self.host_root());
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

    async fn query_model_loaded(&self, model: &str) -> io::Result<bool> {
        let models_url = format!("{}/api/v1/models", self.host_root());
        let response = self
            .client
            .get(&models_url)
            .send()
            .await
            .map_err(|e| io::Error::other(format!("Request failed: {e}")))?;
        if !response.status().is_success() {
            return Err(io::Error::other(format!(
                "Failed to fetch models: {}",
                response.status()
            )));
        }

        let json = response.json::<serde_json::Value>().await.map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("JSON parse error: {e}"))
        })?;
        let models = json.get("models").and_then(|value| value.as_array());
        Ok(models.is_some_and(|entries| {
            entries.iter().any(|entry| {
                let is_requested_model =
                    entry.get("key").and_then(|value| value.as_str()) == Some(model);
                let has_loaded_instances = entry
                    .get("loaded_instances")
                    .and_then(|value| value.as_array())
                    .is_some_and(|instances| !instances.is_empty());
                is_requested_model && has_loaded_instances
            })
        }))
    }

    // Check if a model is already loaded with the same key
    async fn is_model_loaded(&self, model: &str) -> bool {
        self.query_model_loaded(model).await.unwrap_or(false)
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

    async fn fetch_models_response(&self) -> io::Result<LMStudioModelsResponse> {
        let url = format!("{}/api/v1/models", self.host_root());
        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| io::Error::other(format!("Request failed: {e}")))?;

        if !response.status().is_success() {
            return Err(io::Error::other(format!(
                "Failed to fetch models: {}",
                response.status()
            )));
        }

        response
            .json::<LMStudioModelsResponse>()
            .await
            .map_err(|e| {
                io::Error::new(io::ErrorKind::InvalidData, format!("JSON parse error: {e}"))
            })
    }

    // Return the list of models available on the LM Studio server.
    pub async fn fetch_models(&self) -> io::Result<Vec<String>> {
        let response = self.fetch_models_response().await?;
        Ok(response.models.into_iter().map(|m| m.key).collect())
    }

    /// Return model metadata from the LM Studio server.
    pub async fn fetch_model_metadata(&self) -> io::Result<Vec<ModelInfo>> {
        let json = self.fetch_models_response().await?;

        let models = json
            .models
            .into_iter()
            .filter(|model| matches!(model.model_type.as_deref(), None | Some("llm")))
            .enumerate()
            .map(|(index, model)| {
                let context_window = model
                    .loaded_instances
                    .as_ref()
                    .and_then(|instances| {
                        instances
                            .iter()
                            .filter_map(|instance| {
                                instance
                                    .config
                                    .as_ref()
                                    .and_then(|config| config.context_length)
                            })
                            .max()
                    })
                    .or(model.max_context_length);
                // LM Studio reports `capabilities.vision` for models that accept image input, so
                // missing or false values are treated as text-only here.
                let supports_vision = model
                    .capabilities
                    .as_ref()
                    .and_then(|capabilities| capabilities.vision)
                    .unwrap_or(false);
                let trained_for_tool_use = model
                    .capabilities
                    .as_ref()
                    .and_then(|capabilities| capabilities.trained_for_tool_use)
                    .unwrap_or(false);
                let input_modalities = if supports_vision {
                    vec![InputModality::Text, InputModality::Image]
                } else {
                    vec![InputModality::Text]
                };
                let (default_reasoning_level, supported_reasoning_levels) =
                    parse_reasoning_capability(
                        model
                            .capabilities
                            .as_ref()
                            .and_then(|capabilities| capabilities.reasoning.as_ref()),
                    );
                let supports_reasoning =
                    default_reasoning_level.is_some() || !supported_reasoning_levels.is_empty();

                ModelInfo {
                    slug: model.key.clone(),
                    display_name: model
                        .display_name
                        .clone()
                        .unwrap_or_else(|| model.key.clone()),
                    description: model.description,
                    default_reasoning_level,
                    supported_reasoning_levels,
                    shell_type: ConfigShellToolType::Default,
                    visibility: ModelVisibility::List,
                    supported_in_api: true,
                    priority: i32::try_from(index).unwrap_or(i32::MAX),
                    availability_nux: None,
                    upgrade: None,
                    base_instructions: BASE_INSTRUCTIONS.to_string(),
                    model_messages: None,
                    supports_reasoning_summaries: supports_reasoning,
                    default_reasoning_summary: ReasoningSummary::None,
                    support_verbosity: false,
                    default_verbosity: None,
                    apply_patch_tool_type: trained_for_tool_use
                        .then_some(ApplyPatchToolType::Function),
                    truncation_policy: TruncationPolicyConfig::bytes(10_000),
                    supports_parallel_tool_calls: false,
                    context_window,
                    auto_compact_token_limit: None,
                    effective_context_window_percent: 95,
                    experimental_supported_tools: Vec::new(),
                    input_modalities,
                    prefer_websockets: false,
                    used_fallback_model_metadata: false,
                }
            })
            .collect();

        Ok(models)
    }

    pub async fn download_model(&self, model: &str) -> io::Result<()> {
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

        let initial = parse_download_status(&download_status)?;
        let status = initial.status;
        let job_id = initial.job_id;

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

                let mut last_logged = Instant::now() - Duration::from_secs(10);
                let mut attempts = 0u32;

                loop {
                    tokio::time::sleep(DOWNLOAD_POLL_INTERVAL).await;
                    attempts += 1;
                    if attempts > MAX_DOWNLOAD_POLL_ATTEMPTS {
                        eprintln!();
                        return Err(io::Error::other(format!(
                            "Timed out waiting for model '{model}' to download"
                        )));
                    }
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
                    let poll = parse_download_status(&status)?;
                    let status_value = poll.status;
                    let downloaded_bytes = poll.downloaded_bytes;
                    let total_size_bytes = poll.total_size_bytes;

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
                            if let Some(downloaded) = downloaded_bytes {
                                let now = Instant::now();
                                if now.duration_since(last_logged) >= Duration::from_millis(500) {
                                    if let Some(total) = total_size_bytes {
                                        let percent = (downloaded as f64 / total as f64) * 100.0;
                                        eprint!(
                                            "\rDownloading '{model}': {} / {} ({percent:.1}%)",
                                            format_bytes(downloaded),
                                            format_bytes(total)
                                        );
                                    } else {
                                        eprint!(
                                            "\rDownloading '{model}': {}",
                                            format_bytes(downloaded)
                                        );
                                    }
                                    let _ = io::stderr().flush();
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
            .connect_timeout(Duration::from_secs(5))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self {
            client,
            base_url: host_root.into(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct LMStudioModelsResponse {
    models: Vec<LMStudioModel>,
}

#[derive(Debug, Deserialize)]
struct LMStudioModel {
    key: String,
    display_name: Option<String>,
    description: Option<String>,
    #[serde(rename = "type")]
    model_type: Option<String>,
    max_context_length: Option<i64>,
    capabilities: Option<LMStudioCapabilities>,
    loaded_instances: Option<Vec<LMStudioLoadedInstance>>,
}

#[derive(Debug, Deserialize)]
struct LMStudioCapabilities {
    vision: Option<bool>,
    trained_for_tool_use: Option<bool>,
    reasoning: Option<LMStudioReasoningCapability>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum LMStudioReasoningCapability {
    Enabled(bool),
    Options(LMStudioReasoningOptions),
}

#[derive(Debug, Deserialize)]
struct LMStudioReasoningOptions {
    allowed_options: Option<Vec<String>>,
    #[serde(rename = "default")]
    default_option: Option<String>,
}

#[derive(Debug, Deserialize)]
struct LMStudioLoadedInstance {
    config: Option<LMStudioInstanceConfig>,
}

#[derive(Debug, Deserialize)]
struct LMStudioInstanceConfig {
    context_length: Option<i64>,
}

// Poll every 2 seconds in production; use a short interval in tests to avoid slowness.
#[cfg(not(test))]
const DOWNLOAD_POLL_INTERVAL: Duration = Duration::from_secs(2);
#[cfg(test)]
const DOWNLOAD_POLL_INTERVAL: Duration = Duration::from_millis(10);

// Allow ~2 hours of polling (at 2s intervals) before giving up.
const MAX_DOWNLOAD_POLL_ATTEMPTS: u32 = 3600;

struct DownloadStatusResponse {
    status: String,
    job_id: Option<String>,
    downloaded_bytes: Option<u64>,
    total_size_bytes: Option<u64>,
}

fn parse_download_status(json: &serde_json::Value) -> io::Result<DownloadStatusResponse> {
    let status = json["status"]
        .as_str()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing status"))?
        .to_string();
    let job_id = json["job_id"].as_str().map(String::from);
    let downloaded_bytes = json["downloaded_bytes"].as_u64();
    let total_size_bytes = json["total_size_bytes"].as_u64();
    Ok(DownloadStatusResponse {
        status,
        job_id,
        downloaded_bytes,
        total_size_bytes,
    })
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

fn parse_reasoning_effort(option: &str) -> Option<ReasoningEffort> {
    let normalized = option.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "off" | "none" => Some(ReasoningEffort::None),
        "on" => Some(ReasoningEffort::Medium),
        "minimal" => Some(ReasoningEffort::Minimal),
        "low" => Some(ReasoningEffort::Low),
        "medium" => Some(ReasoningEffort::Medium),
        "high" => Some(ReasoningEffort::High),
        "xhigh" => Some(ReasoningEffort::XHigh),
        _ => None,
    }
}

fn parse_reasoning_capability(
    capability: Option<&LMStudioReasoningCapability>,
) -> (Option<ReasoningEffort>, Vec<ReasoningEffortPreset>) {
    let medium_only = (
        Some(ReasoningEffort::Medium),
        vec![ReasoningEffortPreset {
            effort: ReasoningEffort::Medium,
            description: "medium".to_string(),
        }],
    );
    let fallback_presets = vec![
        ReasoningEffort::Low,
        ReasoningEffort::Medium,
        ReasoningEffort::High,
    ]
    .into_iter()
    .map(|effort| ReasoningEffortPreset {
        effort,
        description: format!("{effort}"),
    })
    .collect::<Vec<_>>();
    let fallback = (Some(ReasoningEffort::Medium), fallback_presets);

    let Some(capability) = capability else {
        return (None, Vec::new());
    };

    match capability {
        LMStudioReasoningCapability::Enabled(true) => medium_only,
        LMStudioReasoningCapability::Enabled(false) => (None, Vec::new()),
        LMStudioReasoningCapability::Options(options) => {
            let mut efforts = Vec::new();
            if let Some(allowed_options) = options.allowed_options.as_ref() {
                for option in allowed_options {
                    if let Some(effort) = parse_reasoning_effort(option)
                        && !efforts.contains(&effort)
                    {
                        efforts.push(effort);
                    }
                }
            }

            if efforts.is_empty() {
                return fallback;
            }

            let default_reasoning_level = options
                .default_option
                .as_deref()
                .and_then(parse_reasoning_effort)
                .or_else(|| efforts.first().copied());
            let supported_reasoning_levels = efforts
                .into_iter()
                .map(|effort| ReasoningEffortPreset {
                    effort,
                    description: format!("{effort}"),
                })
                .collect();
            (default_reasoning_level, supported_reasoning_levels)
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::unwrap_used)]
    use super::*;
    use codex_core::spawn::CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR;
    use pretty_assertions::assert_eq;
    use wiremock::Mock;
    use wiremock::MockServer;
    use wiremock::ResponseTemplate;
    use wiremock::matchers::method;
    use wiremock::matchers::path;

    #[tokio::test]
    async fn test_fetch_models_happy_path() {
        if std::env::var(CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR).is_ok() {
            tracing::info!(
                "{} is set; skipping test_fetch_models_happy_path",
                CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR
            );
            return;
        }

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/v1/models"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(
                    serde_json::json!({
                        "models": [
                            {"key": "openai/gpt-oss-20b"},
                        ],
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
        if std::env::var(CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR).is_ok() {
            tracing::info!(
                "{} is set; skipping test_fetch_models_no_data_array",
                CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR
            );
            return;
        }

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/v1/models"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_raw(serde_json::json!({}).to_string(), "application/json"),
            )
            .mount(&server)
            .await;

        let client = LMStudioClient::from_host_root(server.uri());
        let result = client.fetch_models().await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("JSON parse error"));
    }

    #[tokio::test]
    async fn test_fetch_models_server_error() {
        if std::env::var(CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR).is_ok() {
            tracing::info!(
                "{} is set; skipping test_fetch_models_server_error",
                CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR
            );
            return;
        }

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/v1/models"))
            .respond_with(ResponseTemplate::new(500))
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
    async fn test_fetch_model_metadata_filters_and_maps_fields() {
        if std::env::var(CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR).is_ok() {
            tracing::info!(
                "{} is set; skipping test_fetch_model_metadata_filters_and_maps_fields",
                CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR
            );
            return;
        }

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/v1/models"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(
                    serde_json::json!({
                        "models": [
                            {
                                "key": "openai/gpt-oss-20b",
                                "display_name": "GPT-OSS-20B (LM Studio)",
                                "description": "OSS model",
                                "type": "llm",
                                "max_context_length": 100_000,
                                "capabilities": {
                                    "vision": true,
                                    "trained_for_tool_use": true,
                                    "reasoning": {
                                        "allowed_options": ["low", "medium", "high"],
                                        "default": "low"
                                    }
                                },
                                "loaded_instances": [
                                    {
                                        "config": {
                                            "context_length": 90_000
                                        }
                                    }
                                ]
                            },
                            {
                                "key": "embed/text",
                                "display_name": "Embedding",
                                "type": "embedding",
                                "max_context_length": 1024
                            },
                            {
                                "key": "llm/second",
                                "type": "llm",
                                "max_context_length": 4096
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
        let models = client.fetch_model_metadata().await.expect("fetch metadata");

        let expected = vec![
            ModelInfo {
                slug: "openai/gpt-oss-20b".to_string(),
                display_name: "GPT-OSS-20B (LM Studio)".to_string(),
                description: Some("OSS model".to_string()),
                default_reasoning_level: Some(ReasoningEffort::Low),
                supported_reasoning_levels: vec![
                    ReasoningEffortPreset {
                        effort: ReasoningEffort::Low,
                        description: "low".to_string(),
                    },
                    ReasoningEffortPreset {
                        effort: ReasoningEffort::Medium,
                        description: "medium".to_string(),
                    },
                    ReasoningEffortPreset {
                        effort: ReasoningEffort::High,
                        description: "high".to_string(),
                    },
                ],
                shell_type: ConfigShellToolType::Default,
                visibility: ModelVisibility::List,
                supported_in_api: true,
                priority: 0,
                availability_nux: None,
                upgrade: None,
                base_instructions: BASE_INSTRUCTIONS.to_string(),
                model_messages: None,
                supports_reasoning_summaries: true,
                default_reasoning_summary: ReasoningSummary::None,
                support_verbosity: false,
                default_verbosity: None,
                apply_patch_tool_type: Some(ApplyPatchToolType::Function),
                truncation_policy: TruncationPolicyConfig::bytes(10_000),
                supports_parallel_tool_calls: false,
                context_window: Some(90_000),
                auto_compact_token_limit: None,
                effective_context_window_percent: 95,
                experimental_supported_tools: Vec::new(),
                input_modalities: vec![InputModality::Text, InputModality::Image],
                prefer_websockets: false,
                used_fallback_model_metadata: false,
            },
            ModelInfo {
                slug: "llm/second".to_string(),
                display_name: "llm/second".to_string(),
                description: None,
                default_reasoning_level: None,
                supported_reasoning_levels: Vec::new(),
                shell_type: ConfigShellToolType::Default,
                visibility: ModelVisibility::List,
                supported_in_api: true,
                priority: 1,
                availability_nux: None,
                upgrade: None,
                base_instructions: BASE_INSTRUCTIONS.to_string(),
                model_messages: None,
                supports_reasoning_summaries: false,
                default_reasoning_summary: ReasoningSummary::None,
                support_verbosity: false,
                default_verbosity: None,
                apply_patch_tool_type: None,
                truncation_policy: TruncationPolicyConfig::bytes(10_000),
                supports_parallel_tool_calls: false,
                context_window: Some(4096),
                auto_compact_token_limit: None,
                effective_context_window_percent: 95,
                experimental_supported_tools: Vec::new(),
                input_modalities: vec![InputModality::Text],
                prefer_websockets: false,
                used_fallback_model_metadata: false,
            },
        ];

        assert_eq!(models, expected);
    }

    #[tokio::test]
    async fn test_fetch_model_metadata_reasoning_variants() {
        if std::env::var(CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR).is_ok() {
            tracing::info!(
                "{} is set; skipping test_fetch_model_metadata_reasoning_variants",
                CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR
            );
            return;
        }

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/v1/models"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(
                    serde_json::json!({
                        "models": [
                            {
                                "key": "lmstudio-community/qwen3-0.6b",
                                "type": "llm",
                                "capabilities": {
                                    "vision": false,
                                    "trained_for_tool_use": true
                                }
                            },
                            {
                                "key": "qwen/qwen3-0.6b",
                                "type": "llm",
                                "capabilities": {
                                    "vision": false,
                                    "trained_for_tool_use": true,
                                    "reasoning": {
                                        "allowed_options": ["off", "on"],
                                        "default": "on"
                                    }
                                }
                            },
                            {
                                "key": "microsoft/phi-4-mini-reasoning",
                                "type": "llm",
                                "capabilities": {
                                    "vision": false,
                                    "trained_for_tool_use": false,
                                    "reasoning": true
                                }
                            },
                            {
                                "key": "nvidia/nemotron-3-super",
                                "type": "llm",
                                "capabilities": {
                                    "vision": false,
                                    "trained_for_tool_use": true,
                                    "reasoning": {
                                        "allowed_options": ["off", "low", "on"],
                                        "default": "on"
                                    }
                                }
                            },
                            {
                                "key": "test/missing-default",
                                "type": "llm",
                                "capabilities": {
                                    "vision": false,
                                    "trained_for_tool_use": true,
                                    "reasoning": {
                                        "allowed_options": ["off", "on"]
                                    }
                                }
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
        let models = client.fetch_model_metadata().await.expect("fetch metadata");

        let summary = models
            .iter()
            .map(|model| {
                (
                    model.slug.clone(),
                    model.default_reasoning_level,
                    model
                        .supported_reasoning_levels
                        .iter()
                        .map(|preset| preset.effort)
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();

        let expected = vec![
            (
                "lmstudio-community/qwen3-0.6b".to_string(),
                None,
                Vec::<ReasoningEffort>::new(),
            ),
            (
                "qwen/qwen3-0.6b".to_string(),
                Some(ReasoningEffort::Medium),
                vec![ReasoningEffort::None, ReasoningEffort::Medium],
            ),
            (
                "microsoft/phi-4-mini-reasoning".to_string(),
                Some(ReasoningEffort::Medium),
                vec![ReasoningEffort::Medium],
            ),
            (
                "nvidia/nemotron-3-super".to_string(),
                Some(ReasoningEffort::Medium),
                vec![
                    ReasoningEffort::None,
                    ReasoningEffort::Low,
                    ReasoningEffort::Medium,
                ],
            ),
            (
                "test/missing-default".to_string(),
                Some(ReasoningEffort::None),
                vec![ReasoningEffort::None, ReasoningEffort::Medium],
            ),
        ];

        assert_eq!(summary, expected);
    }

    #[tokio::test]
    async fn test_check_server_happy_path() {
        if std::env::var(CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR).is_ok() {
            tracing::info!(
                "{} is set; skipping test_check_server_happy_path",
                CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR
            );
            return;
        }

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/v1/models"))
            .respond_with(ResponseTemplate::new(200))
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
        if std::env::var(CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR).is_ok() {
            tracing::info!(
                "{} is set; skipping test_check_server_error",
                CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR
            );
            return;
        }

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/v1/models"))
            .respond_with(ResponseTemplate::new(404))
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
        if std::env::var(CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR).is_ok() {
            tracing::info!(
                "{} is set; skipping test_load_model_happy_path",
                CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR
            );
            return;
        }

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                serde_json::json!({ "models": [] }).to_string(),
                "application/json",
            ))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/v1/models/load"))
            .respond_with(ResponseTemplate::new(200))
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
        if std::env::var(CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR).is_ok() {
            tracing::info!(
                "{} is set; skipping test_load_model_reuses_loaded_instance",
                CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR
            );
            return;
        }

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/v1/models"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(
                    serde_json::json!({
                        "models": [
                            {
                                "key": "test/test-model",
                                "loaded_instances": [
                                    {
                                        "id": "instance-abc123",
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
        Mock::given(method("POST"))
            .and(path("/api/v1/models/load"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;

        let client = LMStudioClient::from_host_root(server.uri());
        let result = client.load_model("test/test-model").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_is_model_loaded_true() {
        if std::env::var(CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR).is_ok() {
            tracing::info!(
                "{} is set; skipping test_is_model_loaded_true",
                CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR
            );
            return;
        }

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/v1/models"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(
                    serde_json::json!({
                        "models": [
                            {
                                "key": "test/test-model",
                                "loaded_instances": [
                                    {
                                        "id": "instance-abc123",
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
        if std::env::var(CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR).is_ok() {
            tracing::info!(
                "{} is set; skipping test_is_model_loaded_false",
                CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR
            );
            return;
        }

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/v1/models"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(
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
        if std::env::var(CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR).is_ok() {
            tracing::info!(
                "{} is set; skipping test_load_model_error",
                CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR
            );
            return;
        }

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                serde_json::json!({ "models": [] }).to_string(),
                "application/json",
            ))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/v1/models/load"))
            .respond_with(ResponseTemplate::new(500))
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
        if std::env::var(CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR).is_ok() {
            tracing::info!(
                "{} is set; skipping test_download_model_happy_path",
                CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR
            );
            return;
        }

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/v1/models/download"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(
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

        Mock::given(method("GET"))
            .and(path("/api/v1/models/download/status/job-1"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(
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
        if std::env::var(CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR).is_ok() {
            tracing::info!(
                "{} is set; skipping test_download_model_error",
                CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR
            );
            return;
        }

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/v1/models/download"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(
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

        Mock::given(method("GET"))
            .and(path("/api/v1/models/download/status/job-1"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(
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

    #[tokio::test]
    async fn test_download_model_already_downloaded() {
        if std::env::var(CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR).is_ok() {
            tracing::info!(
                "{} is set; skipping test_download_model_already_downloaded",
                CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR
            );
            return;
        }

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/v1/models/download"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(
                    serde_json::json!({
                        "status": "already_downloaded"
                    })
                    .to_string(),
                    "application/json",
                ),
            )
            .mount(&server)
            .await;

        let client = LMStudioClient::from_host_root(server.uri());
        client
            .download_model("openai/gpt-oss-20b")
            .await
            .expect("already_downloaded should succeed");
    }

    #[tokio::test]
    async fn test_download_model_paused() {
        if std::env::var(CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR).is_ok() {
            tracing::info!(
                "{} is set; skipping test_download_model_paused",
                CODEX_SANDBOX_NETWORK_DISABLED_ENV_VAR
            );
            return;
        }

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/v1/models/download"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(
                    serde_json::json!({
                        "status": "paused"
                    })
                    .to_string(),
                    "application/json",
                ),
            )
            .mount(&server)
            .await;

        let client = LMStudioClient::from_host_root(server.uri());
        let result = client.download_model("openai/gpt-oss-20b").await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Model download paused")
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
