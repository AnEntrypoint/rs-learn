pub fn truncate_at_sentence(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars { return text.to_string(); }
    let head: String = text.chars().take(max_chars).collect();
    let bytes = head.as_bytes();
    let mut last_end: Option<usize> = None;
    for (i, &b) in bytes.iter().enumerate() {
        if b == b'.' || b == b'!' || b == b'?' {
            let next = bytes.get(i + 1).copied();
            if next.is_none() || matches!(next, Some(b' ') | Some(b'\n') | Some(b'\t') | Some(b'\r')) {
                last_end = Some(i + 1);
            }
        }
    }
    let cut = last_end.unwrap_or(head.len());
    head[..cut].trim_end().to_string()
}
