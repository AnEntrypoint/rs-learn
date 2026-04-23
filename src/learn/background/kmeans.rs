use rayon::prelude::*;

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
    let dot = crate::simd::dot(a, b);
    let na = crate::simd::dot(a, a).sqrt();
    let nb = crate::simd::dot(b, b).sqrt();
    let denom = (na * nb).max(1e-9);
    1.0 - dot / denom
}

fn normalize(v: &[f32]) -> Vec<f32> {
    let n = crate::simd::dot(v, v).sqrt().max(1e-9);
    v.iter().map(|x| x / n).collect()
}

#[inline]
fn cos_dist_unit(a: &[f32], b: &[f32]) -> f32 {
    1.0 - crate::simd::dot(a, b)
}

pub fn kmeans_plus_plus(vectors: &[Vec<f32>], k: usize, seed: u32) -> Vec<ClusterAssignment> {
    let n = vectors.len();
    if n == 0 || k == 0 { return vec![]; }
    let kk = k.min(n);
    let unit: Vec<Vec<f32>> = vectors.par_iter().map(|v| normalize(v)).collect();
    let mut rng = mulberry32(seed);
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(kk);
    let first = (rng() * n as f32).floor() as usize;
    centroids.push(unit[first.min(n - 1)].clone());
    let mut d2 = vec![0f32; n];
    while centroids.len() < kk {
        let last = centroids.last().unwrap().clone();
        d2.par_iter_mut().enumerate().for_each(|(i, slot)| {
            let dlast = cos_dist_unit(&unit[i], &last);
            if centroids.len() == 1 { *slot = dlast; }
            else if dlast < *slot { *slot = dlast; }
        });
        let sum: f32 = d2.iter().sum();
        if sum == 0.0 { break; }
        let mut r = rng() * sum;
        let mut pick = n - 1;
        for i in 0..n {
            r -= d2[i];
            if r <= 0.0 { pick = i; break; }
        }
        centroids.push(unit[pick].clone());
    }
    let dim = unit[0].len();
    let mut assign = vec![0usize; n];
    for _ in 0..MAX_ITER {
        let new_assign: Vec<usize> = unit.par_iter().map(|v| {
            let mut best = 0usize; let mut bd = f32::INFINITY;
            for (j, c) in centroids.iter().enumerate() {
                let d = cos_dist_unit(v, c);
                if d < bd { bd = d; best = j; }
            }
            best
        }).collect();
        let changed = assign.iter().zip(new_assign.iter()).filter(|(a, b)| a != b).count();
        assign = new_assign;
        let kc = centroids.len();
        let (sums, counts) = unit.par_iter().zip(assign.par_iter()).fold(
            || (vec![0f32; kc * dim], vec![0usize; kc]),
            |(mut s, mut cn), (v, &a)| {
                cn[a] += 1;
                let row = &mut s[a * dim..(a + 1) * dim];
                crate::simd::axpy(1.0, v, row);
                (s, cn)
            },
        ).reduce(
            || (vec![0f32; kc * dim], vec![0usize; kc]),
            |(mut s1, mut c1), (s2, c2)| {
                for i in 0..kc * dim { s1[i] += s2[i]; }
                for i in 0..kc { c1[i] += c2[i]; }
                (s1, c1)
            },
        );
        for j in 0..kc {
            if counts[j] == 0 { continue; }
            let inv = 1.0 / counts[j] as f32;
            for d in 0..dim { centroids[j][d] = sums[j * dim + d] * inv; }
            let nrm = crate::simd::dot(&centroids[j], &centroids[j]).sqrt().max(1e-9);
            for d in 0..dim { centroids[j][d] /= nrm; }
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
        crate::simd::axpy(1.0, &vectors[a.index], &mut sums[a.cluster]);
    }
    for j in 0..kk {
        if counts[j] == 0 { continue; }
        let inv = 1.0 / counts[j] as f32;
        for d in 0..dim { sums[j][d] *= inv; }
    }
    sums
}
