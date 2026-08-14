// Momentum-transfer proposal families.

#[derive(Clone, Debug, Serialize, Deserialize)]
/// A normalized component of a momentum-transfer proposal.
pub enum TComponent {
    /// Uniform density in `t`.
    Uniform,
    /// Density proportional to `exp(slope * t)`.
    Exponential {
        /// Exponential slope.
        slope: f64,
    },
    /// Pole-like density proportional to
    /// $`(\mathit{exchange\_mass}^2 - t)^{-\mathit{power}}`$.
    Pole {
        /// Mass of the exchanged pole.
        exchange_mass: f64,
        /// Power of the pole denominator.
        power: f64,
    },
    /// Piecewise-constant density supplied by a histogram.
    Histogram {
        /// Histogram defining the piecewise density.
        histogram: Histogram,
    },
}

impl TComponent {
    fn sample(&self, low: f64, high: f64, u: f64) -> LadduPhysicsResult<f64> {
        match *self {
            Self::Uniform => Ok(low + u * (high - low)),
            Self::Exponential { slope } => {
                if !slope.is_finite() {
                    return Err(LadduPhysicsError::invalid_value(
                        "exponential t slope",
                        "finite",
                        slope,
                    ));
                }
                if slope.abs() < 1e-10 {
                    return Ok(low + u * (high - low));
                }
                let width = high - low;
                Ok(low + (1.0 + u * (slope * width).exp_m1()).ln() / slope)
            }
            Self::Pole {
                exchange_mass,
                power,
            } => {
                if !exchange_mass.is_finite()
                    || exchange_mass < 0.0
                    || !power.is_finite()
                    || power <= 0.0
                {
                    return Err(LadduPhysicsError::invalid_relation(format!(
                        "pole mass and power must be finite, with nonnegative mass and positive power; got exchange_mass={exchange_mass}, power={power}"
                    )));
                }
                let a = exchange_mass * exchange_mass - high;
                let b = exchange_mass * exchange_mass - low;
                if a <= 0.0 {
                    return Err(LadduPhysicsError::invalid_relation(format!(
                        "pole singularity at {} lies in the physical t interval [{low}, {high}]",
                        exchange_mass * exchange_mass
                    )));
                }
                let x = if (power - 1.0).abs() < 1e-10 {
                    a * (b / a).powf(u)
                } else {
                    let k = 1.0 - power;
                    (a.powf(k) + u * (b.powf(k) - a.powf(k))).powf(1.0 / k)
                };
                Ok(exchange_mass * exchange_mass - x)
            }
            Self::Histogram { ref histogram } => {
                let density = Self::histogram_density(histogram)?;
                density.sample_with_unit(low, high, u).ok_or_else(|| {
                    LadduPhysicsError::invalid_relation(format!(
                        "histogram support does not overlap the physical t interval [{low}, {high}]"
                    ))
                })
            }
        }
    }

    fn density(&self, low: f64, high: f64, t: f64) -> LadduPhysicsResult<f64> {
        match *self {
            Self::Uniform => Ok(1.0 / (high - low)),
            Self::Exponential { slope } => {
                if !slope.is_finite() {
                    return Err(LadduPhysicsError::invalid_value(
                        "exponential t slope",
                        "finite",
                        slope,
                    ));
                }
                if slope.abs() < 1e-10 {
                    return Ok(1.0 / (high - low));
                }
                Ok(slope * (slope * (t - low)).exp() / (slope * (high - low)).exp_m1())
            }
            Self::Pole {
                exchange_mass,
                power,
            } => {
                let a = exchange_mass * exchange_mass - high;
                let b = exchange_mass * exchange_mass - low;
                let x = exchange_mass * exchange_mass - t;
                if a <= 0.0 || power <= 0.0 {
                    return Err(LadduPhysicsError::invalid_relation(format!(
                        "invalid pole component for t interval [{low}, {high}]: exchange_mass={exchange_mass}, power={power}"
                    )));
                }
                let norm = if (power - 1.0).abs() < 1e-10 {
                    (b / a).ln()
                } else {
                    (b.powf(1.0 - power) - a.powf(1.0 - power)) / (1.0 - power)
                };
                Ok(x.powf(-power) / norm)
            }
            Self::Histogram { ref histogram } => {
                let density = Self::histogram_density(histogram)?;
                if density.truncated_total(low, high) <= 0.0 {
                    return Err(LadduPhysicsError::invalid_relation(format!(
                        "histogram support does not overlap the physical t interval [{low}, {high}]"
                    )));
                }
                Ok(density.density_inclusive(low, high, t))
            }
        }
    }

    fn histogram_density(histogram: &Histogram) -> LadduPhysicsResult<PiecewiseDensity> {
        if histogram
            .counts()
            .iter()
            .any(|count| !count.is_finite() || *count < 0.0)
            || !histogram.total_weight().is_finite()
            || histogram.total_weight() <= 0.0
        {
            return Err(LadduPhysicsError::invalid_value(
                "histogram t-proposal counts",
                "finite and nonnegative with positive finite total weight",
                format!("{:?}", histogram.counts()),
            ));
        }
        PiecewiseDensity::from_histogram(histogram).map_err(|_| {
            LadduPhysicsError::invalid_value(
                "histogram t-proposal counts",
                "finite and nonnegative with positive finite total weight",
                format!("{:?}", histogram.counts()),
            )
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
/// Mixture distribution for Mandelstam `t`.
pub struct TDistribution {
    components: Vec<(f64, TComponent)>,
    #[serde(default)]
    t_min: Option<f64>,
    #[serde(default)]
    t_max: Option<f64>,
}

impl TDistribution {
    /// Construct a uniform distribution in `t`.
    pub fn uniform() -> Self {
        Self::mixture([(1.0, TComponent::Uniform)])
    }

    /// Construct an exponential distribution in `t`.
    pub fn exponential(slope: f64) -> Self {
        Self::mixture([(1.0, TComponent::Exponential { slope })])
    }

    /// Construct a pole-like distribution in `t`.
    pub fn pole(exchange_mass: f64, power: f64) -> Self {
        Self::mixture([(
            1.0,
            TComponent::Pole {
                exchange_mass,
                power,
            },
        )])
    }

    /// Construct a histogram-backed distribution in `t`.
    pub fn histogram(histogram: Histogram) -> Self {
        Self::mixture([(1.0, TComponent::Histogram { histogram })])
    }

    /// Construct a weighted mixture of transfer-density components.
    pub fn mixture(components: impl IntoIterator<Item = (f64, TComponent)>) -> Self {
        Self {
            components: components.into_iter().collect(),
            t_min: None,
            t_max: None,
        }
    }

    /// Restrict this proposal to the intersection of these limits and the
    /// event-by-event physical t interval.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when a specified limit is non-finite or
    /// `t_min` is not less than `t_max`.
    ///
    /// # Panics
    ///
    /// Panics only if an option tested as present unexpectedly contains no
    /// value.
    pub fn with_limits(
        mut self,
        t_min: Option<f64>,
        t_max: Option<f64>,
    ) -> LadduPhysicsResult<Self> {
        if t_min.is_some_and(|value| !value.is_finite()) {
            return Err(LadduPhysicsError::invalid_value(
                "t_min",
                "finite when specified",
                t_min.unwrap(),
            ));
        }
        if t_max.is_some_and(|value| !value.is_finite()) {
            return Err(LadduPhysicsError::invalid_value(
                "t_max",
                "finite when specified",
                t_max.unwrap(),
            ));
        }
        if let (Some(t_min), Some(t_max)) = (t_min, t_max)
            && t_max <= t_min
        {
            return Err(LadduPhysicsError::invalid_relation(format!(
                "t limits require t_min < t_max, got [{t_min}, {t_max}]"
            )));
        }
        self.t_min = t_min;
        self.t_max = t_max;
        Ok(self)
    }

    fn normalization(&self) -> LadduPhysicsResult<f64> {
        if self.components.is_empty() {
            return Err(LadduPhysicsError::invalid_length(
                "t-distribution components",
                "at least one",
                0,
            ));
        }
        if self
            .components
            .iter()
            .any(|(weight, _)| !weight.is_finite() || *weight <= 0.0)
        {
            return Err(LadduPhysicsError::invalid_value(
                "t-distribution mixture weights",
                "finite and positive",
                format!(
                    "{:?}",
                    self.components
                        .iter()
                        .map(|(weight, _)| weight)
                        .collect::<Vec<_>>()
                ),
            ));
        }
        let sum: f64 = self.components.iter().map(|(weight, _)| weight).sum();
        Ok(sum)
    }

    fn sample(&self, low: f64, high: f64, rng: &mut ProposalRng) -> LadduPhysicsResult<(f64, f64)> {
        if !low.is_finite() || !high.is_finite() || high <= low {
            return Err(LadduPhysicsError::invalid_relation(format!(
                "physical t interval must have finite bounds with low < high, got [{low}, {high}]"
            )));
        }
        let physical_low = low;
        let physical_high = high;
        let low = self.t_min.map_or(low, |t_min| low.max(t_min));
        let high = self.t_max.map_or(high, |t_max| high.min(t_max));
        if high <= low {
            return Err(LadduPhysicsError::invalid_relation(format!(
                "configured t limits do not overlap the physical interval [{physical_low}, {physical_high}]"
            )));
        }
        let normalization = self.normalization()?;
        let choice = rng.uniform();
        let mut cumulative = 0.0;
        let mut selected = self.components.len() - 1;
        for (index, (weight, _)) in self.components.iter().enumerate() {
            cumulative += weight / normalization;
            if choice < cumulative {
                selected = index;
                break;
            }
        }
        let t = self.components[selected]
            .1
            .sample(low, high, rng.uniform())?;
        let mut density = 0.0;
        for (weight, component) in &self.components {
            density += weight / normalization * component.density(low, high, t)?;
        }
        if !density.is_finite() || density <= 0.0 {
            return Err(LadduPhysicsError::invalid_value(
                "t-proposal density",
                "finite and positive",
                density,
            ));
        }
        Ok((t, density))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
/// Two-to-two scattering proposal based on a selected incoming/outgoing
/// momentum-transfer pairing.
pub struct TwoBodyScattering {
    incoming_edge: String,
    outgoing_edge: String,
    distribution: TDistribution,
}

impl TwoBodyScattering {
    /// Construct a `t`-exchange proposal for the named edge pairing.
    pub fn t_exchange(
        pairing: (impl Into<String>, impl Into<String>),
        distribution: TDistribution,
    ) -> Self {
        Self {
            incoming_edge: pairing.0.into(),
            outgoing_edge: pairing.1.into(),
            distribution,
        }
    }
}

impl From<TwoBodyScattering> for VertexProposal {
    fn from(proposal: TwoBodyScattering) -> Self {
        Self::TwoBodyScattering { proposal }
    }
}

impl TwoBodyScattering {
    /// Propose outgoing two-body scattering kinematics.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when the topology or configured edge
    /// pairing is invalid, the event is outside physical phase space, or the
    /// transfer distribution cannot be sampled.
    pub fn propose(
        &self,
        incoming: &[NamedMomentum<'_>],
        outgoing: &[NamedMass<'_>],
        rng: &mut ProposalRng,
    ) -> LadduPhysicsResult<ProposalResult> {
        if incoming.len() != 2 || outgoing.len() != 2 {
            return Err(LadduPhysicsError::invalid_relation(format!(
                "two-body scattering requires two incoming and two outgoing edges, got {} incoming and {} outgoing",
                incoming.len(),
                outgoing.len()
            )));
        }
        let paired_in = incoming
            .iter()
            .position(|edge| edge.name == self.incoming_edge)
            .ok_or_else(|| {
                LadduPhysicsError::invalid_relation(format!(
                    "unknown incoming t-pairing edge `{}`",
                    self.incoming_edge
                ))
            })?;
        let paired_out = outgoing
            .iter()
            .position(|edge| edge.name == self.outgoing_edge)
            .ok_or_else(|| {
                LadduPhysicsError::invalid_relation(format!(
                    "unknown outgoing t-pairing edge `{}`",
                    self.outgoing_edge
                ))
            })?;
        let total = incoming[0].p4 + incoming[1].p4;
        let root_s = total.m()?;
        let beta = total.beta()?;
        let incoming_com = incoming[paired_in].p4.boost(&(-beta));
        // Invariant masses are best evaluated before the boost. In particular,
        // boosting a massless four-vector can leave a tiny negative m^2 from
        // floating-point cancellation.
        let m1 = incoming[paired_in].p4.m()?;
        let m2 = incoming[1 - paired_in].p4.m()?;
        let m3 = outgoing[paired_out].mass;
        let m4 = outgoing[1 - paired_out].mass;
        let p_in = two_body_momentum(root_s, m1, m2)?;
        let p_out = two_body_momentum(root_s, m3, m4)?;
        if p_in <= 0.0 {
            return Err(LadduPhysicsError::invalid_relation(
                "t exchange is undefined at the incoming threshold",
            ));
        }
        let e1 = (m1 * m1 + p_in * p_in).sqrt();
        let e3 = (m3 * m3 + p_out * p_out).sqrt();
        let center = m1 * m1 + m3 * m3 - 2.0 * e1 * e3;
        let span = 2.0 * p_in * p_out;
        let (t, q_t) = self
            .distribution
            .sample(center - span, center + span, rng)?;
        let cos_theta = ((t - center) / span).clamp(-1.0, 1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
        let phi = 2.0 * PI * rng.uniform();
        let z = incoming_com.vec3().unit()?;
        let seed = if z.z.abs() < 0.9 {
            RealVec3::new(0.0, 0.0, 1.0)
        } else {
            RealVec3::new(1.0, 0.0, 0.0)
        };
        let x = seed.cross(&z).unit()?;
        let y = z.cross(&x);
        let direction = z * cos_theta + x * (sin_theta * phi.cos()) + y * (sin_theta * phi.sin());
        let paired = on_shell(direction, p_out, m3).boost(&beta);
        let other = on_shell(-direction, p_out, m4).boost(&beta);
        let mut result = vec![RealVec4::new(0.0, 0.0, 0.0, 0.0); 2];
        result[paired_out] = paired;
        result[1 - paired_out] = other;
        Ok(ProposalResult {
            outgoing: result,
            weight: 1.0 / (16.0 * PI * root_s * p_in * q_t),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::generation::{AdaptiveTwoBodyDecay, MassProposal, ScalarSource};

    #[test]
    fn proposal_rng_sequence_is_stable() {
        let mut rng = ProposalRng::new(7);
        assert_eq!(
            (0..5).map(|_| rng.next_u64()).collect::<Vec<_>>(),
            [
                7_191_089_600_892_374_487,
                309_689_372_594_955_804,
                16_616_101_746_815_609_346,
                10_753_165_928_301_472_203,
                8_346_079_845_500_723_674,
            ]
        );
    }

    #[test]
    fn isotropic_decay_conserves_momentum_and_mass() {
        let proposal = VertexProposal::isotropic_decay();
        let incoming = [NamedMomentum {
            name: "x",
            p4: RealVec4::new(2.0, 0.3, -0.2, 1.0),
        }];
        let outgoing = [
            NamedMass {
                name: "a",
                mass: 0.2,
            },
            NamedMass {
                name: "b",
                mass: 0.4,
            },
        ];
        let result = proposal
            .propose(&incoming, &outgoing, &mut ProposalRng::new(7))
            .unwrap();
        let sum = result.outgoing[0] + result.outgoing[1];
        for (a, b) in [sum.e, sum.px, sum.py, sum.pz]
            .into_iter()
            .zip([2.0, 0.3, -0.2, 1.0])
        {
            assert!((a - b).abs() < 1e-12);
        }
        assert!((result.outgoing[0].m().unwrap() - 0.2).abs() < 1e-12);
        assert!((result.outgoing[1].m().unwrap() - 0.4).abs() < 1e-12);
        assert!(result.weight > 0.0);
    }

    #[test]
    fn t_mixture_samples_inside_physical_range() {
        let distribution = TDistribution::mixture([
            (1.0, TComponent::Uniform),
            (2.0, TComponent::Exponential { slope: 3.0 }),
            (
                1.0,
                TComponent::Pole {
                    exchange_mass: 1.0,
                    power: 2.0,
                },
            ),
        ]);
        let mut rng = ProposalRng::new(11);
        for _ in 0..100 {
            let (t, density) = distribution.sample(-2.0, -0.1, &mut rng).unwrap();
            assert!((-2.0..=-0.1).contains(&t));
            assert!(density.is_finite() && density > 0.0);
        }
    }

    #[test]
    fn t_distribution_limits_truncate_the_physical_interval() {
        let distribution = TDistribution::uniform()
            .with_limits(Some(-1.25), Some(-0.5))
            .unwrap();
        let mut rng = ProposalRng::new(13);
        for _ in 0..100 {
            let (t, density) = distribution.sample(-2.0, -0.1, &mut rng).unwrap();
            assert!((-1.25..=-0.5).contains(&t));
            assert!((density - 1.0 / 0.75).abs() < 1e-12);
        }
        assert!(
            TDistribution::uniform()
                .with_limits(Some(-0.5), Some(-1.0))
                .is_err()
        );
        assert!(
            distribution
                .sample(-3.0, -2.0, &mut ProposalRng::new(17))
                .is_err()
        );
    }

    #[test]
    fn t_exchange_conserves_momentum_and_is_on_shell() {
        let proposal =
            TwoBodyScattering::t_exchange(("beam", "x"), TDistribution::exponential(2.0));
        let incoming = [
            NamedMomentum {
                name: "beam",
                p4: RealVec4::new(1.5, 0.0, 0.0, 1.0),
            },
            NamedMomentum {
                name: "target",
                p4: RealVec4::new(1.5, 0.0, 0.0, -1.0),
            },
        ];
        let outgoing = [
            NamedMass {
                name: "x",
                mass: 0.5,
            },
            NamedMass {
                name: "r",
                mass: 0.7,
            },
        ];
        let result = proposal
            .propose(&incoming, &outgoing, &mut ProposalRng::new(19))
            .unwrap();
        let before = incoming[0].p4 + incoming[1].p4;
        let after = result.outgoing[0] + result.outgoing[1];
        assert!((before.e - after.e).abs() < 1e-12);
        assert!((before.px - after.px).abs() < 1e-12);
        assert!((before.py - after.py).abs() < 1e-12);
        assert!((before.pz - after.pz).abs() < 1e-12);
        assert!((result.outgoing[0].m().unwrap() - 0.5).abs() < 1e-12);
        assert!((result.outgoing[1].m().unwrap() - 0.7).abs() < 1e-12);
    }

    #[test]
    fn adaptive_decay_preserves_the_phase_space_integral() {
        let incoming = [NamedMomentum {
            name: "parent",
            p4: RealVec4::new(2.0, 0.0, 0.0, 0.0),
        }];
        let outgoing = [
            NamedMass {
                name: "a",
                mass: 0.2,
            },
            NamedMass {
                name: "b",
                mass: 0.4,
            },
        ];
        let adaptive =
            AdaptiveTwoBodyDecay::new(Arc::from([1.0, 2.0, 8.0, 20.0, 8.0, 2.0, 1.0]), 0.2)
                .unwrap();
        let baseline = VertexProposal::isotropic_decay()
            .propose(&incoming, &outgoing, &mut ProposalRng::new(1))
            .unwrap()
            .weight;
        let mut rng = ProposalRng::new(2);
        let samples = 100_000;
        let mean = (0..samples)
            .map(|_| {
                adaptive
                    .propose(&incoming, &outgoing, &mut rng)
                    .unwrap()
                    .weight
            })
            .sum::<f64>()
            / samples as f64;
        assert!((mean / baseline - 1.0).abs() < 0.01);
    }

    #[test]
    fn proposal_failures_use_structured_physics_errors() {
        let empty = TDistribution::mixture([]);
        assert!(matches!(
            empty.normalization(),
            Err(LadduPhysicsError::InvalidLength { .. })
        ));

        assert!(matches!(
            MassProposal::fixed(2.0).propose(0.0, 1.0, &mut ProposalRng::new(0)),
            Err(LadduPhysicsError::InvalidValue { .. })
        ));

        assert!(matches!(
            VertexProposal::isotropic_decay().propose(&[], &[], &mut ProposalRng::new(0)),
            Err(LadduPhysicsError::InvalidRelation { .. })
        ));
    }

    #[test]
    fn histogram_t_component_truncates_to_the_physical_interval() {
        let histogram = Histogram::new(vec![1.0, 3.0], vec![-2.0, -1.0, 0.0]).unwrap();
        let distribution = TDistribution::histogram(histogram);
        let mut rng = ProposalRng::new(31);
        for _ in 0..100 {
            let (t, density) = distribution.sample(-1.5, -0.5, &mut rng).unwrap();
            assert!((-1.5..=-0.5).contains(&t));
            assert!(density.is_finite() && density > 0.0);
        }
    }

    #[test]
    fn scalar_sources_return_values_and_proposal_corrections() {
        let mut rng = ProposalRng::new(37);
        let constant = ScalarSource::constant(3.0).sample(&mut rng).unwrap();
        assert_eq!(constant.value, 3.0);
        assert_eq!(constant.weight, 1.0);

        let uniform = ScalarSource::uniform(-2.0, 4.0).sample(&mut rng).unwrap();
        assert!((-2.0..4.0).contains(&uniform.value));
        assert_eq!(uniform.weight, 6.0);

        let histogram = Histogram::new(vec![1.0, 2.0], vec![0.0, 1.0, 3.0]).unwrap();
        let sampled = ScalarSource::histogram(histogram).sample(&mut rng).unwrap();
        assert!((0.0..3.0).contains(&sampled.value));
        assert!(sampled.weight.is_finite() && sampled.weight > 0.0);
    }

    #[test]
    fn uniform_mass_truncates_to_the_allowed_interval() {
        let proposal = MassProposal::uniform(1.0, 2.0);
        let mut rng = ProposalRng::new(41);
        for _ in 0..100 {
            let result = proposal.propose(1.25, 1.75, &mut rng).unwrap();
            assert!((1.25..1.75).contains(&result.mass));
            assert_eq!(result.weight, 0.5);
        }
    }

    #[test]
    fn continuous_proposals_return_reciprocal_density_weights() {
        let mass = MassProposal::uniform(-1.0, 5.0);
        let mut rng = ProposalRng::new(43);
        for _ in 0..100 {
            let sampled = mass.propose(1.0, 3.0, &mut rng).unwrap();
            let density = mass.density(1.0, 3.0, sampled.mass).unwrap().unwrap();
            assert!((sampled.weight * density - 1.0).abs() < 1e-12);
        }

        let histogram = Histogram::new(vec![1.0, 3.0], vec![0.0, 1.0, 3.0]).unwrap();
        let source = ScalarSource::histogram(histogram.clone());
        for _ in 0..100 {
            let sampled = source.sample(&mut rng).unwrap();
            let density = PiecewiseDensity::from_histogram(&histogram)
                .unwrap()
                .density(0.0, 3.0, sampled.value);
            assert!((sampled.weight * density - 1.0).abs() < 1e-12);
        }
    }
}
