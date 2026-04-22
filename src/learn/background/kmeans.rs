const MAX_ITER: usize = 25;

#[derive(Debug, Clone)]
pub struct ClusterAssignment {
    pub index: usize,
    pub cluster: usize,
}

pub fn mulberry32(seed: u32) -> impl FnMut() -> f32 {
    let mut a: u32 = seed;
    move || {
        a = a.wrapping_add(0x6D2B_79F5);
        let mut t = a;
        t = (t ^ (t >> 15)).wrapping_mul(t | 1);
        t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
        ((t ^ (t >> 14)) as f32) / 4_294_967_296.0
    }
}

pub fn cosine_dist(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0f32; let mut na = 0f32; let mut nb = 0f32;
    for i in 0..a.len() { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    let denom = (na.sqrt() * nb.sqrt()).max(1e-9);
    1.0 - dot / denom
}

pub fn kmeans_plus_plus(vectors: &[Vec<f32>], k: usize, seed: u32) -> Vec<ClusterAssignment> {
    let n = vectors.len();
    if n == 0 || k == 0 { return vec![]; }
    let kk = k.min(n);
    let mut rng = mulberry32(seed);
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(kk);
    let first = (rng() * n as f32).floor() as usize;
    centroids.push(vectors[first.min(n - 1)].clone());
    let mut d2 = vec![0f32; n];
    while centroids.len() < kk {
        let mut sum = 0f32;
        for i in 0..n {
            let mut m = f32::INFINITY;
            for c in &centroids {
                let v = cosine_dist(&vectors[i], c);
                if v < m { m = v; }
            }
            d2[i] = m; sum += m;
        }
        if sum == 0.0 { break; }
        let mut r = rng() * sum;
        let mut pick = n - 1;
        for i in 0..n {
            r -= d2[i];
            if r <= 0.0 { pick = i; break; }
        }
        centroids.push(vectors[pick].clone());
    }
    let dim = vectors[0].len();
    let mut assign = vec![0usize; n];
    for _ in 0..MAX_ITER {
        let mut changed = 0usize;
        for i in 0..n {
            let mut best = 0usize; let mut bd = f32::INFINITY;
            for (j, c) in centroids.iter().enumerate() {
                let d = cosine_dist(&vectors[i], c);
                if d < bd { bd = d; best = j; }
            }
            if assign[i] != best { changed += 1; assign[i] = best; }
        }
        let mut sums: Vec<Vec<f32>> = centroids.iter().map(|_| vec![0f32; dim]).collect();
        let mut counts = vec![0usize; centroids.len()];
        for i in 0..n {
            let a = assign[i]; counts[a] += 1;
            for d in 0..dim { sums[a][d] += vectors[i][d]; }
        }
        for j in 0..centroids.len() {
            if counts[j] == 0 { continue; }
            for d in 0..dim { centroids[j][d] = sums[j][d] / counts[j] as f32; }
        }
        if changed == 0 { break; }
    }
    (0..n).map(|i| ClusterAssignment { index: i, cluster: assign[i] }).collect()
}

pub fn kmeans_centroids(vectors: &[Vec<f32>], assignments: &[ClusterAssignment], k: usize) -> Vec<Vec<f32>> {
    if vectors.is_empty() { return vec![]; }
    let dim = vectors[0].len();
    let kk = k.min(vectors.len());
    let mut sums: Vec<Vec<f32>> = (0..kk).map(|_| vec![0f32; dim]).collect();
    let mut counts = vec![0usize; kk];
    for a in assignments {
        if a.cluster >= kk { continue; }
        counts[a.cluster] += 1;
        for d in 0..dim { sums[a.cluster][d] += vectors[a.index][d]; }
    }
    for j in 0..kk {
        if counts[j] == 0 { continue; }
        for d in 0..dim { sums[j][d] /= counts[j] as f32; }
    }
    sums
}
