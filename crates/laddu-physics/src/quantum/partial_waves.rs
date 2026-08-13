use std::fmt::Display;

use serde::{Deserialize, Serialize};

use crate::{LadduPhysicsError, LadduPhysicsResult};

use super::{J, L, Parity, ParticleProperties, RuleReport, RuleSet, S};

/// A partial wave defined by a total angular momentum, `J`, an orbital angular momentum, `L`, and
/// and intrinsic spin, `S`.
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct PartialWave {
    /// The total angular momentum of the wave
    pub j: J,
    /// The orbital angular momentum of the wave
    pub l: L,
    /// The spin of the wave
    pub s: S,
}
impl PartialWave {
    /// Construct a new partial wave from the given angular momentum quantum numbers.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when `j`, `l`, and `s` violate angular
    /// momentum coupling rules.
    pub fn new(j: J, l: L, s: S) -> LadduPhysicsResult<Self> {
        PartialWave::validate_coupling(j, l, s)?;
        Ok(Self { j, l, s })
    }
    /// Get the spectroscopic label for the wave in the form {2s+1}{l}{j} where l is represented by
    /// its spectroscopic letter equivalent (`S` for `0`, `P` for `1`, etc.).
    pub fn label(&self) -> String {
        let multiplicity = self.s.doubled() + 1;
        format!("{}{}{}", multiplicity, self.l, self.j)
    }
    /// Validate the set of angular momentum quantum numbers which define a partial wave.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when `j` lies outside the range permitted
    /// by `l` and `s` or has incompatible integer/half-integer parity.
    pub fn validate_coupling(j: J, l: L, s: S) -> LadduPhysicsResult<()> {
        let l_twice = 2 * l.value();
        let s_twice = s.doubled();
        let j_twice = j.doubled();
        let min = l_twice.abs_diff(s_twice);
        let max = l_twice + s_twice;
        if j_twice >= min && j_twice <= max && (j_twice - min).is_multiple_of(2) {
            Ok(())
        } else {
            Err(LadduPhysicsError::invalid_relation(
                "j, l, and s must be compatible",
            ))
        }
    }
}

impl Display for PartialWave {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

/// A partial wave together with allowed parity and C-parity, if applicable.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct AllowedPartialWave {
    /// The angular quantum numbers of the wave
    pub wave: PartialWave,
    /// The allowed parity, if applicable
    pub parity: Option<Parity>,
    /// The allowed C-parity, if applicable
    pub c_parity: Option<Parity>,
}

impl AllowedPartialWave {
    /// Take an existing [`PartialWave`] and infer parity and C-parity from its decay products.
    pub fn new(wave: PartialWave, daughters: (&ParticleProperties, &ParticleProperties)) -> Self {
        Self {
            parity: infer_parity(daughters, wave.l),
            c_parity: infer_c_parity(daughters, wave.l, wave.s),
            wave,
        }
    }
}

pub(super) fn infer_parity(
    daughters: (&ParticleProperties, &ParticleProperties),
    l: L,
) -> Option<Parity> {
    Some(daughters.0.parity? * daughters.1.parity? * l.orbital_parity())
}

pub(super) fn infer_c_parity(
    daughters: (&ParticleProperties, &ParticleProperties),
    l: L,
    s: S,
) -> Option<Parity> {
    daughters.0.is_antiparticle_of(daughters.1).then_some(())?;
    let s_doubled = s.doubled();
    if !s_doubled.is_multiple_of(2) {
        return None;
    }
    Some(L::int(l.value() + (s_doubled / 2)).orbital_parity())
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// A generated partial-wave candidate together with its inferred properties and
/// selection-rule report.
pub struct PartialWaveCandidate {
    /// Angular quantum numbers of the candidate.
    pub wave: PartialWave,
    /// Candidate wave plus its channel-dependent inferred parity values.
    pub inferred: AllowedPartialWave,
    /// Detailed outcomes from the configured rules.
    pub report: RuleReport,
}

impl PartialWaveCandidate {
    /// Return whether the candidate passed every enforced rule.
    pub fn is_allowed(&self) -> bool {
        self.report.is_allowed()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Default)]
/// Complete result of scanning a two-body channel for partial waves.
pub struct PartialWaveScan {
    /// All generated candidates, including rejected waves.
    pub candidates: Vec<PartialWaveCandidate>,
    /// Required properties which prevented candidate generation.
    pub missing_inputs: Vec<String>,
}

impl PartialWaveScan {
    /// Iterate over the inferred properties of accepted waves.
    pub fn allowed(&self) -> impl Iterator<Item = &AllowedPartialWave> {
        self.candidates
            .iter()
            .filter(|candidate| candidate.is_allowed())
            .map(|candidate| &candidate.inferred)
    }

    /// Iterate over candidates rejected by at least one enforced rule.
    pub fn rejected(&self) -> impl Iterator<Item = &PartialWaveCandidate> {
        self.candidates
            .iter()
            .filter(|candidate| !candidate.is_allowed())
    }

    /// Consume the scan and collect its accepted waves.
    pub fn into_allowed(self) -> Vec<AllowedPartialWave> {
        self.candidates
            .into_iter()
            .filter_map(|candidate| {
                if candidate.is_allowed() {
                    Some(candidate.inferred)
                } else {
                    None
                }
            })
            .collect()
    }
}

/// Configuration for generating and filtering allowed two-body partial waves.
///
/// `SelectionRules` combines a maximum orbital angular momentum with a
/// [`RuleSet`]. Candidate waves are generated from angular-momentum coupling
/// and are then filtered by the enabled rules.
///
/// The generated waves satisfy
/// $`S \in |j_a - j_b|, \ldots, j_a + j_b`$
/// and
/// $`J \in |L - S|, \ldots, L + S`$,
/// with $`0 \le L \le L_\text{max}`$.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct SelectionRules {
    /// Conservation and symmetry rules used to filter candidate waves.
    ///
    /// Angular-momentum compatibility is handled by
    /// [`SelectionRules::allowed_partial_waves`]. The [`RuleSet`] applies
    /// additional checks such as parity, charge, isospin, flavor quantum
    /// numbers, $`C`$-parity, $`G`$-parity, and identical-particle symmetry.
    pub rules: RuleSet,
    /// Maximum orbital angular momentum $`L_\text{max}`$ considered when
    /// generating candidate partial waves.
    ///
    /// The solver scans all integer values
    /// $`L = 0, 1, \ldots, L_\text{max}`$.
    pub max_l: L,
}

impl Default for SelectionRules {
    fn default() -> Self {
        Self::strong(L::int(6))
    }
}

impl SelectionRules {
    /// Construct a partial-wave scanner from a rule set and maximum orbital
    /// angular momentum.
    pub fn new(rules: RuleSet, max_l: L) -> Self {
        Self { rules, max_l }
    }

    /// Construct a scanner which applies only angular-momentum coupling.
    pub fn angular(max_l: L) -> Self {
        Self::new(RuleSet::angular(), max_l)
    }

    /// Construct a scanner configured for electromagnetic decays.
    pub fn electromagnetic(max_l: L) -> Self {
        Self::new(RuleSet::electromagnetic(), max_l)
    }

    /// Construct a scanner configured for weak decays.
    pub fn weak(max_l: L) -> Self {
        Self::new(RuleSet::weak(), max_l)
    }

    /// Construct a scanner configured for strong decays.
    pub fn strong(max_l: L) -> Self {
        Self::new(RuleSet::strong(), max_l)
    }
    /// Return all possible coupled total spins from two daughter spins.
    ///
    /// Given daughter spins $`j_a`$ and $`j_b`$, this returns
    /// $`S = |j_a - j_b|, |j_a - j_b| + 1, \ldots, j_a + j_b`$.
    ///
    /// Internally angular momenta are stored as doubled values, so the returned
    /// sequence advances by two in the doubled representation.
    pub fn coupled_spins(a: J, b: J) -> Vec<S> {
        a.coupled_with(b)
    }

    /// Generate all candidates and retain detailed reports for accepted and
    /// rejected waves.
    pub fn scan_partial_waves(
        &self,
        parent: &ParticleProperties,
        daughters: (&ParticleProperties, &ParticleProperties),
    ) -> PartialWaveScan {
        let mut missing_inputs = Vec::new();

        let Some(parent_j) = parent.spin else {
            missing_inputs.push("parent.spin".to_string());
            return PartialWaveScan {
                candidates: Vec::new(),
                missing_inputs,
            };
        };

        let Some(ja) = daughters.0.spin else {
            missing_inputs.push("daughter_a.spin".to_string());
            return PartialWaveScan {
                candidates: Vec::new(),
                missing_inputs,
            };
        };

        let Some(jb) = daughters.1.spin else {
            missing_inputs.push("daughter_b.spin".to_string());
            return PartialWaveScan {
                candidates: Vec::new(),
                missing_inputs,
            };
        };

        let mut candidates = Vec::new();

        for s in Self::coupled_spins(ja, jb) {
            for l_raw in 0..=self.max_l.value() {
                let l = L::int(l_raw);

                let Ok(wave) = PartialWave::new(parent_j, l, s) else {
                    continue;
                };

                let report = self.rules.evaluate(parent, daughters, l, s);
                let inferred = AllowedPartialWave::new(wave, daughters);

                candidates.push(PartialWaveCandidate {
                    wave,
                    inferred,
                    report,
                });
            }
        }

        PartialWaveScan {
            candidates,
            missing_inputs,
        }
    }

    /// Generate all allowed two-body partial waves for a parent and two
    /// daughters.
    ///
    /// The parent spin is interpreted as the total angular momentum $`J`$ of
    /// the resonance. The daughter spins are coupled to possible total-spin
    /// values $`S`$, and each $`S`$ is combined with orbital angular momenta
    /// $`L = 0, 1, \ldots, L_\text{max}`$.
    ///
    /// A candidate wave is kept when:
    ///
    /// 1. $`L`$ and $`S`$ can couple to the parent $`J`$.
    /// 2. The enabled [`RuleSet`] checks do not reject it.
    ///
    /// Returns an empty vector if the parent spin or either daughter spin is
    /// unknown.
    ///
    /// The returned [`AllowedPartialWave`] includes the underlying
    /// [`PartialWave`] together with channel-dependent inferred quantum numbers,
    /// such as final-state parity and, when meaningful, $`C`$-parity.
    pub fn allowed_partial_waves(
        &self,
        parent: &ParticleProperties,
        daughters: (&ParticleProperties, &ParticleProperties),
    ) -> Vec<AllowedPartialWave> {
        self.scan_partial_waves(parent, daughters).into_allowed()
    }
}
