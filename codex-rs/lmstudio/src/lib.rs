mod client;

pub use client::LMStudioClient;
use codex_core::config::Config;

// Default OSS model to use when `--oss` is passed without an explicit `-m`.
pub const DEFAULT_OSS_MODEL: &str = "openai/gpt-oss-20b";

// Prepare the local OSS environment when `--oss` is selected.
//
// - Esnures a local LM Studio server is reachable.
// - Checks if the model exists locally and downloads it if missing.
pub async fn ensure_oss_ready(config: &Config) -> std::io::Result<()> {
    let model: &str = config.model.as_ref();

    // Verify local LM Studio is reachable.
    let lmstudio_client = LMStudioClient::try_from_provider(config).await?;

    match lmstudio_client.fetch_models().await {
        Ok(models) => {
            if !models.iter().any(|m| m == DEFAULT_OSS_MODEL) {
                eprintln!("Downloading model: {}", DEFAULT_OSS_MODEL);

                let status = std::process::Command::new("lms")
                    .args(["get", "--yes", DEFAULT_OSS_MODEL])
                    .stdout(std::process::Stdio::inherit())
                    .stderr(std::process::Stdio::inherit())
                    .status()
                    .map_err(|e| {
                        std::io::Error::new(
                            std::io::ErrorKind::Other,
                            format!("Failed to execute 'lms get --yes {DEFAULT_OSS_MODEL}': {e}"),
                        )
                    })?;

                if !status.success() {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("lms command failed with status: {status}"),
                    ));
                }
                tracing::info!("Successfully downloaded model '{model}'");
            }
        }
        Err(err) => {
            // Not fatal; higher layers may still proceed and surface errors later.
            tracing::warn!("Failed to query local models from LM Studio: {}.", err);
        }
    }

    Ok(())
}
