/// Deterministic, portable random-number stream for generation proposals.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProposalRng {
    state: u64,
}

impl ProposalRng {
    /// Construct a proposal stream from a reproducible seed.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Draw the next uniformly distributed 64-bit integer.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Draw a floating-point value strictly between zero and one.
    pub fn uniform(&mut self) -> f64 {
        const SCALE: f64 = 1.0 / ((1_u64 << 53) as f64);
        ((self.next_u64() >> 11) as f64 + 0.5) * SCALE
    }

    pub(super) fn isotropic_direction(&mut self) -> RealVec3 {
        let cos_theta = 2.0 * self.uniform() - 1.0;
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
        let phi = 2.0 * PI * self.uniform();
        RealVec3::new(sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta)
    }
}
