use crate::attention::{Context as AttnContext, Subgraph, SubgraphNode};

pub(super) fn attention_hint(ctx: &AttnContext, subgraph: &Subgraph) -> String {
    let valid: Vec<(usize, &SubgraphNode)> = subgraph.nodes.iter().enumerate()
        .filter(|(_, n)| n.embedding.as_ref().map(|e| e.len() == 768).unwrap_or(false))
        .collect();
    if valid.is_empty() || ctx.weights.is_empty() { return String::new(); }
    let n = valid.len();
    let mut mean = vec![0.0f32; n];
    for head in &ctx.weights {
        if head.len() != n { return String::new(); }
        for (i, w) in head.iter().enumerate() { mean[i] += *w; }
    }
    let h = ctx.weights.len() as f32;
    for v in mean.iter_mut() { *v /= h; }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|a, b| mean[*b].partial_cmp(&mean[*a]).unwrap_or(std::cmp::Ordering::Equal));
    let top: Vec<String> = idx.iter().take(3).map(|i| format!("- {} (attn={:.3})", valid[*i].1.id, mean[*i])).collect();
    format!("\n\nTop-attended context:\n{}", top.join("\n"))
}

pub(super) fn implicit_quality_from(latency_ms: u64, grounding: Option<f32>, confidence: f32) -> f64 {
    let latency_score = (5000.0f64 - (latency_ms as f64).min(5000.0)) / 5000.0;
    let conf = confidence.clamp(0.0, 1.0) as f64;
    match grounding {
        None => (0.35 * latency_score + 0.15 * conf + 0.50 * 0.5).clamp(0.0, 1.0),
        Some(g) => {
            let ground = g.clamp(0.0, 1.0) as f64;
            let q = 0.35 * latency_score + 0.50 * ground + 0.15 * conf;
            if ground < 0.15 { return (q * 0.5).clamp(0.0, 1.0); }
            q.clamp(0.0, 1.0)
        }
    }
}
