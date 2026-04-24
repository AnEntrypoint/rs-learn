use super::*;

impl Router {
    pub(super) fn pack(&self) -> Vec<u8> {
        let parts: [&[f32]; 15] = [
            &self.w.v, &self.w.u, &self.w.uh, &self.w.bh, &self.w.bz,
            &self.heads.model, &self.heads.model_b, &self.heads.ctx, &self.heads.ctx_b,
            &self.heads.temp, &self.heads.temp_b, &self.heads.top_p, &self.heads.top_p_b,
            &self.heads.conf, &self.heads.conf_b,
        ];
        let n: usize = parts.iter().map(|p| p.len()).sum();
        let mut flat = Vec::with_capacity(n);
        for p in parts { flat.extend_from_slice(p); }
        cast_slice::<f32, u8>(&flat).to_vec()
    }

    pub(super) fn unpack(&mut self, bytes: &[u8]) -> Result<()> {
        if bytes.len() % 4 != 0 { return Err(anyhow!("router: blob not f32-aligned")); }
        let mut flat = vec![0f32; bytes.len() / 4];
        cast_slice_mut::<f32, u8>(&mut flat).copy_from_slice(bytes);
        let nt = self.targets.len();
        let sizes = [RANK*IN, DIM*RANK, DIM*DIM, DIM, DIM, nt*DIM, nt, CTX_BUCKETS*DIM, CTX_BUCKETS, DIM, 1, DIM, 1, DIM, 1];
        let total: usize = sizes.iter().sum();
        if flat.len() != total { return Err(anyhow!("router: blob size mismatch want {} got {}", total, flat.len())); }
        let mut o = 0usize;
        let mut take = |n: usize| -> Vec<f32> { let s = flat[o..o+n].to_vec(); o += n; s };
        self.w = Weights { v: take(RANK*IN), u: take(DIM*RANK), uh: take(DIM*DIM), bh: take(DIM), bz: take(DIM) };
        self.heads = Heads {
            model: take(nt*DIM), model_b: take(nt),
            ctx: take(CTX_BUCKETS*DIM), ctx_b: take(CTX_BUCKETS),
            temp: take(DIM), temp_b: take(1),
            top_p: take(DIM), top_p_b: take(1),
            conf: take(DIM), conf_b: take(1),
        };
        self.trained = true;
        Ok(())
    }

    pub async fn save(&mut self) -> Result<i64> {
        self.version += 1;
        let blob = self.pack();
        let meta = json!({ "dim": DIM, "sparsity": SPARSITY, "rank": RANK, "inputDim": IN, "nTargets": self.targets.len() });
        self.store.save_router_weights(self.version, &blob, "fastgrnn", &meta).await?;
        self.trained = true;
        Ok(self.version)
    }

    pub async fn load(&mut self) -> Result<Option<i64>> {
        let Some(row) = self.store.load_latest_router_weights().await? else { return Ok(None) };
        self.unpack(&row.blob)?;
        self.version = row.version;
        Ok(Some(row.version))
    }
}
