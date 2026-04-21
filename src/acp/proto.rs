use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const PROTOCOL_VERSION: u32 = 1;

#[derive(Serialize)]
pub struct Request<'a> {
    pub jsonrpc: &'static str,
    pub id: u64,
    pub method: &'a str,
    pub params: Value,
}

#[derive(Serialize)]
pub struct Response {
    pub jsonrpc: &'static str,
    pub id: Value,
    pub result: Value,
}

#[derive(Deserialize, Debug)]
pub struct RpcMessage {
    #[allow(dead_code)]
    pub jsonrpc: Option<String>,
    pub id: Option<Value>,
    pub method: Option<String>,
    pub params: Option<Value>,
    pub result: Option<Value>,
    pub error: Option<Value>,
}

pub fn parse_json_payload(text: &str) -> Option<Value> {
    let trimmed = text.trim();
    let body = strip_fence(trimmed);
    let bytes = body.as_bytes();
    let start = find_start(bytes)?;
    let (open, close) = (bytes[start], if bytes[start] == b'{' { b'}' } else { b']' });
    let end = balance(bytes, start, open, close)?;
    serde_json::from_slice::<Value>(&bytes[start..=end]).ok()
}

fn strip_fence(t: &str) -> &str {
    let Some(first) = t.find("```") else { return t; };
    let after = &t[first + 3..];
    let after = after.strip_prefix("json").unwrap_or(after);
    let after = after.trim_start_matches(|c: char| c == '\n' || c == '\r' || c == ' ');
    match after.find("```") { Some(end) => &after[..end], None => after }
}

fn find_start(b: &[u8]) -> Option<usize> {
    let c = b.iter().position(|&x| x == b'{');
    let a = b.iter().position(|&x| x == b'[');
    match (c, a) {
        (Some(x), Some(y)) => Some(x.min(y)),
        (Some(x), None) => Some(x),
        (None, Some(y)) => Some(y),
        _ => None,
    }
}

fn balance(b: &[u8], start: usize, open: u8, close: u8) -> Option<usize> {
    let (mut depth, mut in_str, mut esc) = (0i32, false, false);
    for i in start..b.len() {
        let c = b[i];
        if in_str {
            if esc { esc = false; }
            else if c == b'\\' { esc = true; }
            else if c == b'"' { in_str = false; }
            continue;
        }
        match c {
            b'"' => in_str = true,
            x if x == open => depth += 1,
            x if x == close => { depth -= 1; if depth == 0 { return Some(i); } }
            _ => {}
        }
    }
    None
}
