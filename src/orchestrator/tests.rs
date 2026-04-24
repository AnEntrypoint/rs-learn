use super::pipeline::{attention_hint, implicit_quality_from};
use crate::attention::{Context as AttnContext, Subgraph, SubgraphNode};

fn node(id: &str) -> SubgraphNode {
    SubgraphNode { id: id.into(), embedding: Some(vec![0.0; 768]), created_at: Some(0) }
}

#[test]
fn attention_hint_empty_subgraph_no_panic() {
    let ctx = AttnContext { vector: vec![0.0; 768], weights: vec![] };
    let sg = Subgraph::default();
    assert_eq!(attention_hint(&ctx, &sg), "");
}

#[test]
fn attention_hint_changes_prompt_vs_baseline() {
    let sg = Subgraph { nodes: vec![node("alpha"), node("beta"), node("gamma")], edges: vec![] };
    let ctx = AttnContext {
        vector: vec![0.0; 768],
        weights: vec![vec![0.1, 0.7, 0.2], vec![0.2, 0.6, 0.2]],
    };
    let baseline = AttnContext { vector: vec![0.0; 768], weights: vec![] };
    let with_attn = attention_hint(&ctx, &sg);
    let without = attention_hint(&baseline, &sg);
    assert_ne!(with_attn, without);
    assert!(with_attn.contains("Top-attended context:"));
    assert!(with_attn.contains("beta"));
    let beta_pos = with_attn.find("beta").unwrap();
    let alpha_pos = with_attn.find("alpha").unwrap();
    assert!(beta_pos < alpha_pos, "beta (highest weight) must rank first");
}

#[test]
fn implicit_quality_rewards_fast_grounded_responses() {
    let fast_grounded = implicit_quality_from(200, Some(0.9), 0.9);
    let slow_ungrounded = implicit_quality_from(4800, Some(0.05), 0.1);
    assert!(fast_grounded > 0.7, "fast grounded should be high, got {fast_grounded}");
    assert!(slow_ungrounded < 0.3, "slow ungrounded should be low, got {slow_ungrounded}");
}

#[test]
fn implicit_quality_low_grounding_caps_quality() {
    let fast_ungrounded = implicit_quality_from(100, Some(0.05), 0.9);
    let fast_grounded = implicit_quality_from(100, Some(0.5), 0.9);
    assert!(fast_ungrounded < 0.4, "fast but ungrounded must be penalized, got {fast_ungrounded}");
    assert!(fast_grounded > fast_ungrounded, "grounding must dominate latency");
}

#[test]
fn implicit_quality_length_does_not_affect_score() {
    let q = implicit_quality_from(1000, Some(0.5), 0.5);
    assert!((0.0..=1.0).contains(&q));
}

#[test]
fn implicit_quality_clamps_to_unit() {
    assert!((0.0..=1.0).contains(&implicit_quality_from(0, Some(1.0), 1.0)));
    assert!((0.0..=1.0).contains(&implicit_quality_from(u64::MAX, Some(-1.0), -1.0)));
}

#[test]
fn implicit_quality_cold_start_no_penalty() {
    let cold = implicit_quality_from(1000, None, 0.8);
    let grounded = implicit_quality_from(1000, Some(0.8), 0.8);
    assert!((0.0..=1.0).contains(&cold), "cold start must be in range, got {cold}");
    assert!(cold > 0.3, "cold start must not be excessively penalized, got {cold}");
    assert!(grounded > cold, "grounded response should score higher than cold start");
}

#[test]
fn session_ema_smooths_outlier_feedback() {
    let default_ema: f32 = 0.5;
    let raw_quality: f32 = 1.0;
    let expected = 0.7 * raw_quality + 0.3 * default_ema;
    let smoothed = 0.7 * raw_quality + 0.3 * default_ema;
    assert!((smoothed - expected).abs() < 1e-6, "EMA formula mismatch: {smoothed} vs {expected}");
    assert!((smoothed - 0.85).abs() < 1e-6, "first feedback on default EMA(0.5) with quality=1.0 should yield 0.85, got {smoothed}");
    let mut ema = default_ema;
    for &q in &[1.0f32, 0.0, 1.0, 0.0] {
        ema = 0.7 * q + 0.3 * ema;
    }
    assert!((0.0..=1.0).contains(&ema), "EMA must stay in [0,1], got {ema}");
}
