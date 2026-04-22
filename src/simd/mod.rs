use pulp::Arch;

#[inline]
pub fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut s = 0.0f32;
    for i in 0..n { s += a[i] * b[i]; }
    s
}

#[inline]
pub fn axpy_scalar(alpha: f32, x: &[f32], y: &mut [f32]) {
    let n = x.len().min(y.len());
    for i in 0..n { y[i] += alpha * x[i]; }
}

#[inline]
pub fn matvec_scalar(w: &[f32], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) {
    for r in 0..rows {
        let base = r * cols;
        out[r] = dot_scalar(&w[base..base + cols], x);
    }
}

#[inline]
pub fn outer_add_scalar(alpha: f32, u: &[f32], v: &[f32], w: &mut [f32]) {
    let rows = u.len();
    let cols = v.len();
    for r in 0..rows {
        let a = alpha * u[r];
        let row = &mut w[r * cols..(r + 1) * cols];
        for c in 0..cols { row[c] += a * v[c]; }
    }
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let arch = Arch::new();
    arch.dispatch(|| {
        let n = a.len().min(b.len());
        let chunks = n / 8;
        let mut acc = [0.0f32; 8];
        for c in 0..chunks {
            let ai = c * 8;
            for k in 0..8 { acc[k] += a[ai + k] * b[ai + k]; }
        }
        let mut s = acc.iter().sum::<f32>();
        for i in chunks * 8..n { s += a[i] * b[i]; }
        s
    })
}

pub fn axpy(alpha: f32, x: &[f32], y: &mut [f32]) {
    let arch = Arch::new();
    arch.dispatch(|| {
        let n = x.len().min(y.len());
        let chunks = n / 8;
        for c in 0..chunks {
            let ai = c * 8;
            for k in 0..8 { y[ai + k] += alpha * x[ai + k]; }
        }
        for i in chunks * 8..n { y[i] += alpha * x[i]; }
    });
}

pub fn matvec(w: &[f32], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) {
    let arch = Arch::new();
    arch.dispatch(|| {
        for r in 0..rows {
            let base = r * cols;
            let row = &w[base..base + cols];
            let chunks = cols / 8;
            let mut acc = [0.0f32; 8];
            for c in 0..chunks {
                let ai = c * 8;
                for k in 0..8 { acc[k] += row[ai + k] * x[ai + k]; }
            }
            let mut s = acc.iter().sum::<f32>();
            for i in chunks * 8..cols { s += row[i] * x[i]; }
            out[r] = s;
        }
    });
}

pub fn outer_add(alpha: f32, u: &[f32], v: &[f32], w: &mut [f32]) {
    let arch = Arch::new();
    arch.dispatch(|| {
        let rows = u.len();
        let cols = v.len();
        for r in 0..rows {
            let a = alpha * u[r];
            let row = &mut w[r * cols..(r + 1) * cols];
            let chunks = cols / 8;
            for c in 0..chunks {
                let ai = c * 8;
                for k in 0..8 { row[ai + k] += a * v[ai + k]; }
            }
            for i in chunks * 8..cols { row[i] += a * v[i]; }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rnd(seed: u32, n: usize) -> Vec<f32> {
        let mut a = seed;
        (0..n).map(|_| {
            a = a.wrapping_mul(1664525).wrapping_add(1013904223);
            ((a >> 8) as f32 / 16777216.0) - 0.5
        }).collect()
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
    }

    #[test]
    fn dot_parity_768() {
        let a = rnd(1, 768);
        let b = rnd(2, 768);
        let diff = (dot(&a, &b) - dot_scalar(&a, &b)).abs();
        assert!(diff < 1e-3, "dot diff {}", diff);
    }

    #[test]
    fn axpy_parity_768() {
        let x = rnd(3, 768);
        let mut y1 = rnd(4, 768);
        let mut y2 = y1.clone();
        axpy(0.37, &x, &mut y1);
        axpy_scalar(0.37, &x, &mut y2);
        assert!(max_abs_diff(&y1, &y2) < 1e-5);
    }

    #[test]
    fn matvec_parity_96x768() {
        let w = rnd(5, 96 * 768);
        let x = rnd(6, 768);
        let mut o1 = vec![0.0f32; 96];
        let mut o2 = vec![0.0f32; 96];
        matvec(&w, 96, 768, &x, &mut o1);
        matvec_scalar(&w, 96, 768, &x, &mut o2);
        assert!(max_abs_diff(&o1, &o2) < 1e-3);
    }

    #[test]
    fn outer_add_parity_2x768() {
        let u = rnd(7, 2);
        let v = rnd(8, 768);
        let mut w1 = vec![0.0f32; 2 * 768];
        let mut w2 = vec![0.0f32; 2 * 768];
        outer_add(0.1, &u, &v, &mut w1);
        outer_add_scalar(0.1, &u, &v, &mut w2);
        assert!(max_abs_diff(&w1, &w2) < 1e-5);
    }

    #[test]
    fn odd_length_handled() {
        let a = rnd(9, 17);
        let b = rnd(10, 17);
        let diff = (dot(&a, &b) - dot_scalar(&a, &b)).abs();
        assert!(diff < 1e-5);
    }
}
