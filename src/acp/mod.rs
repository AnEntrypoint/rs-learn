mod proto;

pub use proto::parse_json_payload;

use crate::errors::{LlmError, Result};
use proto::{Request, Response, RpcMessage, PROTOCOL_VERSION};
use serde_json::{json, Value};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{oneshot, Mutex};
use tokio::time::timeout;

const DEFAULT_TIMEOUT_MS: u64 = 120_000;

pub struct AcpClient {
    cmd: String,
    args: Vec<String>,
    inner: Arc<Mutex<Option<Session>>>,
    turn_lock: Arc<Mutex<()>>,
}

struct Session {
    child: Child,
    stdin: ChildStdin,
    session_id: String,
    next_id: AtomicU64,
    pending: Arc<Mutex<std::collections::HashMap<u64, oneshot::Sender<Value>>>>,
    chunks: Arc<Mutex<Option<Vec<String>>>>,
}

impl AcpClient {
    pub fn from_env() -> Result<Self> {
        let raw = std::env::var("RS_LEARN_ACP_COMMAND")
            .map_err(|_| LlmError::Process("RS_LEARN_ACP_COMMAND not set".into()))?;
        let mut parts = raw.split_whitespace();
        let cmd = parts.next()
            .ok_or_else(|| LlmError::Process("RS_LEARN_ACP_COMMAND empty".into()))?
            .to_string();
        let mut args: Vec<String> = parts.map(String::from).collect();
        if let Ok(extra) = std::env::var("RS_LEARN_ACP_ARGS") {
            match serde_json::from_str::<Vec<String>>(&extra) {
                Ok(a) => args.extend(a),
                Err(_) => args.extend(extra.split_whitespace().map(String::from)),
            }
        }
        Ok(Self {
            cmd,
            args,
            inner: Arc::new(Mutex::new(None)),
            turn_lock: Arc::new(Mutex::new(())),
        })
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

    async fn ensure_session(&self) -> Result<()> {
        let mut guard = self.inner.lock().await;
        if guard.as_mut().map_or(false, |s| s.child.try_wait().ok().flatten().is_none()) {
            return Ok(());
        }
        *guard = Some(self.spawn_session().await?);
        Ok(())
    }

    async fn spawn_session(&self) -> Result<Session> {
        let spawn_cmd = self.resolve_cmd();
        let mut builder = Command::new(&spawn_cmd);
        builder
            .args(&self.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            builder.creation_flags(0x08000000);
        }
        let mut child = builder
            .spawn()
            .map_err(|e| LlmError::Process(format!("spawn {}: {}", spawn_cmd, e)))?;
        let stdin = child.stdin.take().ok_or_else(|| LlmError::Process("stdin".into()))?;
        let stdout = child.stdout.take().ok_or_else(|| LlmError::Process("stdout".into()))?;
        let stderr = child.stderr.take().ok_or_else(|| LlmError::Process("stderr".into()))?;

        let pending: Arc<Mutex<std::collections::HashMap<u64, oneshot::Sender<Value>>>> =
            Arc::new(Mutex::new(std::collections::HashMap::new()));
        let chunks: Arc<Mutex<Option<Vec<String>>>> = Arc::new(Mutex::new(None));
        spawn_reader(stdout, pending.clone(), chunks.clone());
        spawn_stderr_drain(stderr);

        let mut sess = Session {
            child,
            stdin,
            session_id: String::new(),
            next_id: AtomicU64::new(1),
            pending,
            chunks,
        };
        let init_params = json!({
            "protocolVersion": PROTOCOL_VERSION,
            "clientCapabilities": { "fs": { "readTextFile": true, "writeTextFile": true } }
        });
        sess.call("initialize", init_params, 30_000).await?;
        let cwd = std::env::current_dir().map(|p| p.to_string_lossy().into_owned()).unwrap_or_default();
        let new_sess = sess.call("session/new", json!({ "cwd": cwd, "mcpServers": [] }), 30_000).await?;
        sess.session_id = new_sess.get("sessionId")
            .and_then(|v| v.as_str())
            .ok_or_else(|| LlmError::Validation("missing sessionId".into()))?
            .to_string();
        Ok(sess)
    }

    pub async fn generate(&self, system: &str, user: &str, timeout_ms: u64) -> Result<Value> {
        let _gate = crate::llm_gate::acquire().await
            .map_err(|e| LlmError::Process(format!("llm semaphore closed: {e}")))?;
        let _lock = self.turn_lock.lock().await;
        let prompt = if system.is_empty() {
            format!("{}\n\nRespond with ONLY a JSON object. No preamble. No code fence.", user)
        } else {
            format!("{}\n\n---\n\n{}\n\nRespond with ONLY a JSON object. No preamble. No code fence.", system, user)
        };
        let tmo = if timeout_ms == 0 { DEFAULT_TIMEOUT_MS } else { timeout_ms };
        for attempt in 0..2u8 {
            match self.run_turn(&prompt, tmo).await {
                Ok(text) => {
                    return parse_json_payload(&text).ok_or_else(|| {
                        LlmError::Validation(format!("ACP returned non-JSON (len={})", text.len()))
                    });
                }
                Err(e) => {
                    self.kill_session().await;
                    if attempt == 1 {
                        return Err(match e {
                            LlmError::Process(_) | LlmError::Aborted(_) => e,
                            other => LlmError::Transient(format!("ACP turn failed: {}", other)),
                        });
                    }
                }
            }
        }
        Err(LlmError::Transient("unreachable".into()))
    }

    async fn run_turn(&self, prompt_text: &str, timeout_ms: u64) -> Result<String> {
        self.ensure_session().await?;
        let mut guard = self.inner.lock().await;
        let sess = guard.as_mut().ok_or_else(|| LlmError::Process("no session".into()))?;
        let buf: Vec<String> = Vec::new();
        *sess.chunks.lock().await = Some(buf);
        let params = json!({
            "sessionId": sess.session_id,
            "prompt": [{ "type": "text", "text": prompt_text }]
        });
        let res = sess.call("session/prompt", params, timeout_ms).await;
        let collected = sess.chunks.lock().await.take().unwrap_or_default();
        res?;
        Ok(collected.join(""))
    }

    async fn kill_session(&self) {
        if let Some(mut s) = self.inner.lock().await.take() {
            let _ = s.child.start_kill();
        }
    }

    pub async fn close(self) { self.kill_session().await; }
}

impl Session {
    async fn call(&mut self, method: &str, params: Value, timeout_ms: u64) -> Result<Value> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        self.pending.lock().await.insert(id, tx);
        let req = Request { jsonrpc: "2.0", id, method, params };
        let mut line = serde_json::to_vec(&req)?;
        line.push(b'\n');
        self.stdin.write_all(&line).await.map_err(|e| LlmError::Process(format!("write: {}", e)))?;
        self.stdin.flush().await.map_err(|e| LlmError::Process(format!("flush: {}", e)))?;
        match timeout(Duration::from_millis(timeout_ms), rx).await {
            Ok(Ok(v)) => Ok(v),
            Ok(Err(_)) => Err(LlmError::Process("ACP process exited mid-turn".into())),
            Err(_) => Err(LlmError::Timeout(format!("ACP {} timed out after {}ms", method, timeout_ms))),
        }
    }
}

fn spawn_reader(
    stdout: tokio::process::ChildStdout,
    pending: Arc<Mutex<std::collections::HashMap<u64, oneshot::Sender<Value>>>>,
    chunks: Arc<Mutex<Option<Vec<String>>>>,
) {
    tokio::spawn(async move {
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();
        loop {
            line.clear();
            match reader.read_line(&mut line).await {
                Ok(0) | Err(_) => break,
                Ok(_) => {}
            }
            let Ok(msg) = serde_json::from_str::<RpcMessage>(line.trim()) else { continue };
            if let (Some(id), Some(result)) = (&msg.id, msg.result.clone()) {
                if let Some(id_u64) = id.as_u64() {
                    if let Some(tx) = pending.lock().await.remove(&id_u64) { let _ = tx.send(result); }
                    continue;
                }
            }
            if let (Some(id), Some(err)) = (&msg.id, &msg.error) {
                if let Some(id_u64) = id.as_u64() {
                    if let Some(tx) = pending.lock().await.remove(&id_u64) {
                        let _ = tx.send(serde_json::json!({ "__acp_error": err }));
                    }
                }
            }
            if let (Some(method), Some(params)) = (&msg.method, &msg.params) {
                handle_notification(method, params, &chunks, &pending, &msg.id).await;
            }
        }
    });
}

async fn handle_notification(
    method: &str,
    params: &Value,
    chunks: &Arc<Mutex<Option<Vec<String>>>>,
    _pending: &Arc<Mutex<std::collections::HashMap<u64, oneshot::Sender<Value>>>>,
    _id: &Option<Value>,
) {
    if method == "session/update" {
        let upd = params.get("update");
        let kind = upd.and_then(|u| u.get("sessionUpdate")).and_then(|v| v.as_str());
        if kind == Some("agent_message_chunk") {
            let text = upd
                .and_then(|u| u.get("content"))
                .and_then(|c| if c.get("type").and_then(|t| t.as_str()) == Some("text") { c.get("text") } else { None })
                .and_then(|t| t.as_str());
            if let Some(t) = text {
                if let Some(buf) = chunks.lock().await.as_mut() { buf.push(t.to_string()); }
            }
        }
    }
}

fn spawn_stderr_drain(stderr: tokio::process::ChildStderr) {
    tokio::spawn(async move {
        let mut reader = BufReader::new(stderr);
        let mut line = String::new();
        let debug = std::env::var("RS_LEARN_DEBUG_ACP").is_ok();
        loop {
            line.clear();
            match reader.read_line(&mut line).await {
                Ok(0) | Err(_) => break,
                Ok(_) => if debug { eprintln!("[acp] {}", line.trim_end()); }
            }
        }
    });
}

#[allow(dead_code)]
fn suppress_unused(r: Response) -> Response { r }
