use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use serde::{Deserialize, Serialize};
use std::io::{self, BufRead, Write};
use std::path::PathBuf;

#[derive(Deserialize)]
struct Request {
    id: u64,
    text: String,
}

#[derive(Serialize)]
struct Response {
    id: u64,
    embedding: Vec<f32>,
}

#[derive(Serialize)]
struct ErrorResponse {
    id: u64,
    error: String,
}

#[derive(Serialize)]
struct Ready {
    ready: bool,
    dim: u32,
    model: &'static str,
}

fn emit<T: Serialize>(stdout: &mut io::StdoutLock<'_>, value: &T) -> io::Result<()> {
    let s = serde_json::to_string(value).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    stdout.write_all(s.as_bytes())?;
    stdout.write_all(b"\n")?;
    stdout.flush()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut opts = InitOptions::new(EmbeddingModel::NomicEmbedTextV15);
    if let Ok(dir) = std::env::var("RS_LEARN_EMBED_CACHE") {
        opts = opts.with_cache_dir(PathBuf::from(dir));
    }
    let mut model = TextEmbedding::try_new(opts)?;

    let stdout = io::stdout();
    let mut out = stdout.lock();
    emit(
        &mut out,
        &Ready {
            ready: true,
            dim: 768,
            model: "nomic-embed-text-v1.5",
        },
    )?;

    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let req: Request = match serde_json::from_str(trimmed) {
            Ok(r) => r,
            Err(e) => {
                emit(
                    &mut out,
                    &ErrorResponse {
                        id: 0,
                        error: format!("parse: {}", e),
                    },
                )?;
                continue;
            }
        };
        match model.embed(vec![req.text.clone()], None) {
            Ok(mut embs) => {
                if let Some(emb) = embs.pop() {
                    emit(
                        &mut out,
                        &Response {
                            id: req.id,
                            embedding: emb,
                        },
                    )?;
                } else {
                    emit(
                        &mut out,
                        &ErrorResponse {
                            id: req.id,
                            error: "empty embedding".into(),
                        },
                    )?;
                }
            }
            Err(e) => {
                emit(
                    &mut out,
                    &ErrorResponse {
                        id: req.id,
                        error: format!("embed: {}", e),
                    },
                )?;
            }
        }
    }
    Ok(())
}
