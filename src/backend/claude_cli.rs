use crate::acp::parse_json_payload;
use crate::errors::{LlmError, Result};
use serde_json::Value;
use std::process::Stdio;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::Command;
use tokio::time::timeout;

const DEFAULT_TIMEOUT_MS: u64 = 120_000;
const DEFAULT_MODEL: &str = "haiku";

pub struct ClaudeCliClient {
    cmd: String,
    model: String,
    extra_args: Vec<String>,
    plugin_dir: Option<String>,
}

impl ClaudeCliClient {
    pub fn from_env() -> Result<Self> {
        let cmd = std::env::var("RS_LEARN_CLAUDE_CLI")
            .unwrap_or_else(|_| "claude".into());
        let model = std::env::var("RS_LEARN_CLAUDE_MODEL")
            .unwrap_or_else(|_| DEFAULT_MODEL.into());
        let extra_args = std::env::var("RS_LEARN_CLAUDE_ARGS")
            .ok()
            .and_then(|s| serde_json::from_str::<Vec<String>>(&s).ok())
            .unwrap_or_default();
        let plugin_dir = std::env::var("RS_LEARN_CLAUDE_PLUGIN_DIR").ok();
        Ok(Self { cmd, model, extra_args, plugin_dir })
    }

    fn resolve_cmd(&self) -> String {
        if cfg!(windows) {
            let lower = self.cmd.to_lowercase();
            if !(lower.ends_with(".exe") || lower.ends_with(".cmd") || lower.ends_with(".bat")) {
                return format!("{}.cmd", self.cmd);
            }
        }
        self.cmd.clone()
    }

    pub async fn generate(&self, system: &str, user: &str, timeout_ms: u64) -> Result<Value> {
        let prompt = if system.is_empty() {
            format!("{}\n\nRespond with ONLY a JSON object. No preamble. No code fence.", user)
        } else {
            format!("{}\n\n---\n\n{}\n\nRespond with ONLY a JSON object. No preamble. No code fence.", system, user)
        };
        let tmo = if timeout_ms == 0 { DEFAULT_TIMEOUT_MS } else { timeout_ms };

        let spawn_cmd = self.resolve_cmd();
        let mut cli_args: Vec<String> = vec![
            "-p".into(),
            "--model".into(), self.model.clone(),
            "--output-format".into(), "json".into(),
            "--dangerously-skip-permissions".into(),
            "--no-session-persistence".into(),
        ];
        if let Some(pd) = &self.plugin_dir {
            cli_args.push("--plugin-dir".into());
            cli_args.push(pd.clone());
        }
        cli_args.extend(self.extra_args.iter().cloned());

        let mut child = Command::new(&spawn_cmd)
            .args(&cli_args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| LlmError::Process(format!("spawn {}: {}", spawn_cmd, e)))?;

        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(prompt.as_bytes()).await
                .map_err(|e| LlmError::Process(format!("stdin write: {}", e)))?;
            let _ = stdin.shutdown().await;
        }

        let mut stdout = child.stdout.take().ok_or_else(|| LlmError::Process("stdout".into()))?;
        let mut stderr = child.stderr.take().ok_or_else(|| LlmError::Process("stderr".into()))?;

        let collect = async {
            let mut out = Vec::new();
            let mut err = Vec::new();
            let (ro, re) = tokio::join!(stdout.read_to_end(&mut out), stderr.read_to_end(&mut err));
            ro.map_err(|e| LlmError::Process(format!("stdout read: {}", e)))?;
            re.map_err(|e| LlmError::Process(format!("stderr read: {}", e)))?;
            let status = child.wait().await.map_err(|e| LlmError::Process(format!("wait: {}", e)))?;
            Ok::<_, LlmError>((status, out, err))
        };

        let (status, stdout_bytes, stderr_bytes) = timeout(Duration::from_millis(tmo), collect)
            .await
            .map_err(|_| LlmError::Timeout(format!("claude -p timed out after {}ms", tmo)))??;

        if !status.success() {
            let err_text = String::from_utf8_lossy(&stderr_bytes);
            return Err(LlmError::Process(format!(
                "claude -p exit {:?}: {}",
                status.code(),
                err_text.trim()
            )));
        }

        let stdout_text = String::from_utf8_lossy(&stdout_bytes);
        let envelope: Value = serde_json::from_str(stdout_text.trim())
            .map_err(|e| LlmError::Validation(format!("claude -p non-JSON stdout: {} (body={})", e, stdout_text.chars().take(400).collect::<String>())))?;

        if envelope.get("is_error").and_then(|v| v.as_bool()) == Some(true) {
            return Err(LlmError::Process(format!("claude -p is_error: {}", envelope)));
        }
        let result_text = envelope.get("result")
            .and_then(|v| v.as_str())
            .ok_or_else(|| LlmError::Validation(format!("claude -p missing .result field: {}", envelope)))?;

        parse_json_payload(result_text).ok_or_else(|| {
            LlmError::Validation(format!("claude -p .result not JSON (len={}): {}", result_text.len(), result_text.chars().take(200).collect::<String>()))
        })
    }
}
