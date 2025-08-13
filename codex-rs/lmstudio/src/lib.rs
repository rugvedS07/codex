mod client;

pub use client::LMStudioClient;
use codex_core::config::Config;
use std::path::Path;

// Default OSS model to use when `--oss` is passed without an explicit `-m`.
pub const DEFAULT_OSS_MODEL: &str = "openai/gpt-oss-20b";

// Find the lms binary, checking fallback paths if not in PATH
fn find_lms_binary() -> std::io::Result<String> {
    find_lms_binary_with_home_dir(None)
}

fn find_lms_binary_with_home_dir(home_dir: Option<&str>) -> std::io::Result<String> {
    // First try 'lms' in PATH
    if which::which("lms").is_ok() {
        return Ok("lms".to_string());
    }

    // Platform-specific fallback paths
    let home = match home_dir {
        Some(dir) => dir.to_string(),
        None => {
            #[cfg(unix)]
            {
                std::env::var("HOME").unwrap_or_default()
            }
            #[cfg(windows)]
            {
                std::env::var("USERPROFILE").unwrap_or_default()
            }
        }
    };

    #[cfg(unix)]
    let fallback_path = format!("{home}/.lmstudio/bin/lms");

    #[cfg(windows)]
    let fallback_path = format!("{home}/.lmstudio/bin/lms.exe");

    if Path::new(&fallback_path).exists() {
        Ok(fallback_path)
    } else {
        Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "LM Studio not found. Please install LM Studio from https://lmstudio.ai/",
        ))
    }
}

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
                eprintln!("Downloading model: {DEFAULT_OSS_MODEL}");

                let lms_binary = find_lms_binary()?;
                let status = std::process::Command::new(&lms_binary)
                    .args(["get", "--yes", DEFAULT_OSS_MODEL])
                    .stdout(std::process::Stdio::inherit())
                    .stderr(std::process::Stdio::inherit())
                    .status()
                    .map_err(|e| {
                        std::io::Error::other(format!(
                            "Failed to execute '{lms_binary} get --yes {DEFAULT_OSS_MODEL}': {e}"
                        ))
                    })?;

                if !status.success() {
                    return Err(std::io::Error::other(format!(
                        "lms command failed with status: {status}"
                    )));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_lms_binary() {
        let result = find_lms_binary();

        match result {
            Ok(_) => {
                // lms was found in PATH - that's fine
            }
            Err(e) => {
                // Expected error when LM Studio not installed
                assert!(e.to_string().contains("LM Studio not found"));
            }
        }
    }

    #[test]
    fn test_find_lms_binary_with_mock_home() {
        // Test fallback path construction without touching env vars
        #[cfg(unix)]
        {
            let result = find_lms_binary_with_home_dir(Some("/test/home"));
            if let Err(e) = result {
                assert!(e.to_string().contains("LM Studio not found"));
            }
        }

        #[cfg(windows)]
        {
            let result = find_lms_binary_with_home_dir(Some("C:\\test\\home"));
            if let Err(e) = result {
                assert!(e.to_string().contains("LM Studio not found"));
            }
        }
    }
}
