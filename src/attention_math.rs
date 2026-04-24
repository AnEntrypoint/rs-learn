use std::time::{SystemTime, UNIX_EPOCH};

pub(super) fn mulberry32(seed: u32) -> impl FnMut() -> f32 {
    let mut a: u32 = seed;
    move || {
        a = a.wrapping_add(0x6D2B79F5);
        let mut t = a;
        t = (t ^ (t >> 15)).wrapping_mul(t | 1);
        t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
        ((t ^ (t >> 14)) as f32) / 4294967296.0
    }
}

pub(super) fn rand_matrix(rows: usize, cols: usize, rng: &mut impl FnMut() -> f32, scale: f32) -> Vec<f32> {
    let mut m = vec![0.0f32; rows * cols];
    for v in m.iter_mut() { *v = (rng() * 2.0 - 1.0) * scale; }
    m
}

pub(super) fn layer_norm(x: &[f32]) -> Vec<f32> {
    let n = x.len() as f32;
    let mean: f32 = x.iter().sum::<f32>() / n;
    let var: f32 = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    let inv = 1.0 / (var + 1e-5).sqrt();
    x.iter().map(|v| (v - mean) * inv).collect()
}

pub(super) fn now_ms_f32() -> f32 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as f32).unwrap_or(0.0)
}
