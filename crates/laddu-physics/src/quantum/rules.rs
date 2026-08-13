use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::quantum::{L, ParticleProperties, S};

mod evaluators;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
/// A conservation, symmetry, or classification rule that can be applied to a
/// two-body decay.
pub enum RuleKind {
    /// Enforce intrinsic parity conservation.
    ///
    /// For a two-body final state this checks
    /// $`P_\text{parent} = P_a P_b (-1)^L`$.
    Parity,

    /// Enforce total isospin coupling.
    ///
    /// This checks whether the two daughter isospins can couple to the parent
    /// isospin:
    /// $`I_\text{parent} \in |I_a - I_b|, \ldots, I_a + I_b`$.
    Isospin,
    /// Enforce conservation of the isospin projection $`I_3`$.
    ///
    /// This checks $`I_{3,\text{parent}} = I_{3,a} + I_{3,b}`$.
    IsospinProjection,
    /// Enforce charge-conjugation parity conservation when applicable.
    ///
    /// This is only meaningful for states with a defined $`C`$ eigenvalue and
    /// final states that can be interpreted as $`C`$ eigenstates, such as
    /// suitable particle-antiparticle combinations.
    CParity,
    /// Enforce G-parity conservation when applicable.
    ///
    /// This is mainly useful for light-quark isospin multiplets where
    /// $`G`$-parity is defined. It should not be enabled blindly for arbitrary
    /// hadrons.
    GParity,
    /// Enforce electric charge conservation.
    ///
    /// This checks $`Q_\text{parent} = Q_a + Q_b`$.
    Charge,
    /// Enforce strangeness conservation.
    ///
    /// This checks $`S_\text{parent} = S_a + S_b`$.
    ///
    /// Strong and electromagnetic interactions conserve strangeness; weak
    /// interactions generally do not.
    Strangeness,
    /// Enforce charm conservation.
    ///
    /// This checks $`C_\text{parent} = C_a + C_b`$, where $`C`$ here denotes
    /// charm quantum number, not charge conjugation.
    Charm,
    /// Enforce bottomness conservation.
    ///
    /// This checks $`B'_\text{parent} = B'_a + B'_b`$, where $`B'`$ denotes
    /// bottomness, not baryon number.
    Bottomness,
    /// Enforce topness conservation.
    ///
    /// This checks $`T_\text{parent} = T_a + T_b`$.
    Topness,
    /// Enforce baryon-number conservation.
    ///
    /// This checks $`B_\text{parent} = B_a + B_b`$.
    BaryonNumber,
    /// Enforce electron-family lepton-number conservation.
    ///
    /// This checks $`L_e(\text{parent}) = L_e(a) + L_e(b)`$.
    ElectronLeptonNumber,
    /// Enforce muon-family lepton-number conservation.
    ///
    /// This checks $`L_\mu(\text{parent}) = L_\mu(a) + L_\mu(b)`$.
    MuonLeptonNumber,
    /// Enforce tau-family lepton-number conservation.
    ///
    /// This checks $`L_\tau(\text{parent}) = L_\tau(a) + L_\tau(b)`$.
    TauLeptonNumber,
    /// Enforce total lepton-number conservation.
    ///
    /// This checks $`L_\text{parent} = L_a + L_b`$, where
    /// $`L = L_e + L_\mu + L_\tau`$.
    ///
    /// This is independent of the individual lepton-family checks. If both this
    /// and the family-specific checks are enabled, all enabled checks must pass.
    LeptonNumber,
    /// Enforce exchange-symmetry constraints for identical final-state
    /// particles when enough information is available.
    ///
    /// At minimum, this is useful for cases such as identical spin-zero bosons,
    /// where only even $`L`$ is allowed.
    IdenticalParticleSymmetry,

    /// Optional diagnostic/classification rule, not part of strong-decay conservation.
    ConventionalMesonJpc,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
/// Controls whether and how a rule contributes to the acceptance decision.
pub enum RuleMode {
    /// Reject candidates for which the rule fails.
    Enforce,
    /// Skip the rule, optionally recording why it was disabled.
    Ignore {
        /// Optional explanation for ignoring the rule.
        reason: Option<String>,
    },
    /// Evaluate and report the rule without rejecting the candidate.
    DiagnoseOnly {
        /// Optional explanation for retaining the rule as a diagnostic.
        reason: Option<String>,
    },
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
/// Determines how an enforced rule treats missing particle properties.
pub enum UnknownPolicy {
    /// Current behavior: missing information does not reject the channel.
    Allow,
    /// Missing information makes the rule fail.
    Reject,
    /// Missing information does not reject, but appears in the report.
    Warn,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
/// The evaluation mode and missing-input policy associated with one rule.
pub struct RulePolicy {
    /// Whether the rule is enforced, ignored, or diagnostic.
    pub mode: RuleMode,
    /// How missing inputs are handled when the rule is enforced.
    pub unknown: UnknownPolicy,
}

impl RulePolicy {
    /// Create an enforced policy which allows unknown inputs.
    pub fn enforce() -> Self {
        Self {
            mode: RuleMode::Enforce,
            unknown: UnknownPolicy::Allow,
        }
    }

    /// Create an enforced policy which rejects unknown inputs.
    pub fn enforce_strict() -> Self {
        Self {
            mode: RuleMode::Enforce,
            unknown: UnknownPolicy::Reject,
        }
    }

    /// Create an enforced policy with an explicit missing-input policy.
    pub fn enforce_with_unknown_policy(unknown: UnknownPolicy) -> Self {
        Self {
            mode: RuleMode::Enforce,
            unknown,
        }
    }

    /// Create an ignored policy and record a reason.
    pub fn ignore(reason: impl Into<String>) -> Self {
        Self {
            mode: RuleMode::Ignore {
                reason: Some(reason.into()),
            },
            unknown: UnknownPolicy::Allow,
        }
    }

    /// Create an ignored policy without recording a reason.
    pub fn ignore_without_reason() -> Self {
        Self {
            mode: RuleMode::Ignore { reason: None },
            unknown: UnknownPolicy::Allow,
        }
    }

    /// Create a diagnostic-only policy and record a reason.
    pub fn diagnose_only(reason: impl Into<String>) -> Self {
        Self {
            mode: RuleMode::DiagnoseOnly {
                reason: Some(reason.into()),
            },
            unknown: UnknownPolicy::Warn,
        }
    }

    /// Create a diagnostic-only policy without recording a reason.
    pub fn diagnose_only_without_reason() -> Self {
        Self {
            mode: RuleMode::DiagnoseOnly { reason: None },
            unknown: UnknownPolicy::Warn,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
/// The result of applying one configured rule to a candidate partial wave.
pub enum RuleOutcome {
    /// The rule was evaluated and satisfied.
    Pass {
        /// Human-readable explanation of the successful check.
        message: String,
    },
    /// The rule was evaluated and rejected the candidate.
    Fail {
        /// Human-readable explanation of the failure.
        message: String,
    },
    /// Required inputs were absent, and the policy allowed the candidate.
    UnknownAllowed {
        /// Names of the particle properties that were unavailable.
        missing: Vec<String>,
        /// Human-readable explanation of the incomplete check.
        message: String,
    },
    /// Required inputs were absent and the policy requested a warning.
    Warning {
        /// Names of the particle properties that were unavailable.
        missing: Vec<String>,
        /// Human-readable explanation of the incomplete check.
        message: String,
    },
    /// The rule was intentionally not evaluated.
    Ignored {
        /// Optional explanation supplied when the rule was ignored.
        reason: Option<String>,
    },
    /// The rule was evaluated for information but did not affect acceptance.
    Diagnostic {
        /// Whether the check passed, or `None` when inputs were unavailable.
        passed: Option<bool>,
        /// Optional explanation supplied when diagnostic mode was selected.
        reason: Option<String>,
        /// Human-readable result of the diagnostic check.
        message: String,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
/// The outcome of one named rule in a [`RuleReport`].
pub struct RuleCheck {
    /// Rule that was evaluated.
    pub rule: RuleKind,
    /// Result produced under the rule's configured policy.
    pub outcome: RuleOutcome,
}

impl RuleCheck {
    /// Return whether this check rejects the candidate.
    pub fn is_failure(&self) -> bool {
        matches!(self.outcome, RuleOutcome::Fail { .. })
    }

    /// Return whether this check produced a missing-input warning.
    pub fn is_warning(&self) -> bool {
        matches!(self.outcome, RuleOutcome::Warning { .. })
    }

    /// Return whether missing inputs were accepted silently.
    pub fn is_unknown_allowed(&self) -> bool {
        matches!(self.outcome, RuleOutcome::UnknownAllowed { .. })
    }

    /// Return whether this rule was ignored.
    pub fn is_ignored(&self) -> bool {
        matches!(self.outcome, RuleOutcome::Ignored { .. })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Default, Serialize, Deserialize)]
/// Detailed results from evaluating a [`RuleSet`] against one candidate.
pub struct RuleReport {
    /// Individual checks, ordered by [`RuleKind`].
    pub checks: Vec<RuleCheck>,
}

impl RuleReport {
    /// Return whether no enforced rule rejected the candidate.
    pub fn is_allowed(&self) -> bool {
        self.checks.iter().all(|check| !check.is_failure())
    }

    /// Iterate over checks that rejected the candidate.
    pub fn failures(&self) -> impl Iterator<Item = &RuleCheck> {
        self.checks.iter().filter(|check| check.is_failure())
    }

    /// Iterate over missing-input warnings.
    pub fn warnings(&self) -> impl Iterator<Item = &RuleCheck> {
        self.checks.iter().filter(|check| check.is_warning())
    }

    /// Iterate over checks whose missing inputs were allowed.
    pub fn unknowns(&self) -> impl Iterator<Item = &RuleCheck> {
        self.checks
            .iter()
            .filter(|check| check.is_unknown_allowed())
    }

    /// Iterate over rules which were intentionally ignored.
    pub fn ignored(&self) -> impl Iterator<Item = &RuleCheck> {
        self.checks.iter().filter(|check| check.is_ignored())
    }

    /// Retrieve the outcome for a particular rule, if it was configured.
    pub fn outcome(&self, rule: RuleKind) -> Option<&RuleOutcome> {
        self.checks
            .iter()
            .find(|check| check.rule == rule)
            .map(|check| &check.outcome)
    }

    /// Return whether at least one check rejected the candidate.
    pub fn has_failures(&self) -> bool {
        self.failures().next().is_some()
    }

    /// Return whether at least one check allowed unknown inputs.
    pub fn has_unknowns(&self) -> bool {
        self.unknowns().next().is_some()
    }

    /// Return whether at least one configured rule was ignored.
    pub fn has_ignored(&self) -> bool {
        self.ignored().next().is_some()
    }

    /// Return the number of configured rules in the report.
    pub fn len(&self) -> usize {
        self.checks.len()
    }

    /// Return whether the report contains no checks.
    pub fn is_empty(&self) -> bool {
        self.checks.is_empty()
    }
}

/// A collection of selection rules for testing whether a two-body
/// decay channel is allowed.
///
/// Each rule enables one conservation or symmetry check. Each enabled rule is associated with a
/// [`RulePolicy`] which dictates how permissively it should be applied to the given particles.
///
/// # Notes
/// The default angular policy doesn't actually enforce any rules, as angular momentum conservation
/// and coupling rules are handled by other methods.
///
/// All constructors assume a permissive enforcement policy, i.e. if a property is unknown for one
/// or more particles involved, that check is skipped.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Default, Serialize, Deserialize)]
pub struct RuleSet {
    policies: BTreeMap<RuleKind, RulePolicy>,
}
impl RuleSet {
    /// Construct a rule set with no non-angular selection rules enabled.
    ///
    /// This is useful when only the angular-momentum coupling constraints should
    /// be applied:
    /// $`S \in |j_a - j_b|, \ldots, j_a + j_b`$
    /// and
    /// $`J \in |L - S|, \ldots, L + S`$.
    pub fn angular() -> Self {
        Self::default()
    }

    /// Construct a rule set appropriate for ordinary strong two-body decays.
    ///
    /// This enables parity, isospin, isospin projection, electric charge,
    /// flavor quantum numbers, baryon number, and identical-particle exchange
    /// symmetry.
    ///
    /// Charge-conjugation parity and G-parity are left disabled because they
    /// are only meaningful for certain channels and should be enabled
    /// explicitly when applicable.
    pub fn strong() -> Self {
        Self::angular()
            .enforce(RuleKind::Parity)
            .enforce(RuleKind::Isospin)
            .enforce(RuleKind::IsospinProjection)
            .enforce(RuleKind::Charge)
            .enforce(RuleKind::Strangeness)
            .enforce(RuleKind::Charm)
            .enforce(RuleKind::Bottomness)
            .enforce(RuleKind::Topness)
            .enforce(RuleKind::BaryonNumber)
            .enforce(RuleKind::IdenticalParticleSymmetry)
    }

    /// Construct a rule set appropriate for electromagnetic two-body decays.
    ///
    /// This enables parity, electric charge, flavor quantum numbers, baryon
    /// number, isospin-projection conservation, and identical-particle exchange
    /// symmetry.
    ///
    /// Total isospin is not enabled because electromagnetic interactions break
    /// isospin symmetry.
    pub fn electromagnetic() -> Self {
        Self::angular()
            .enforce(RuleKind::Parity)
            .enforce(RuleKind::IsospinProjection)
            .enforce(RuleKind::Charge)
            .enforce(RuleKind::Strangeness)
            .enforce(RuleKind::Charm)
            .enforce(RuleKind::Bottomness)
            .enforce(RuleKind::Topness)
            .enforce(RuleKind::BaryonNumber)
            .enforce(RuleKind::IdenticalParticleSymmetry)
    }

    /// Construct a rule set appropriate for weak two-body decays.
    ///
    /// This enables electric charge, baryon number, individual lepton-family
    /// numbers, total lepton number, and identical-particle exchange symmetry.
    ///
    /// Parity, isospin, strangeness, charm, bottomness, and topness are not
    /// enabled because weak interactions can violate or change them.
    pub fn weak() -> Self {
        Self::angular()
            .enforce(RuleKind::Charge)
            .enforce(RuleKind::BaryonNumber)
            .enforce(RuleKind::ElectronLeptonNumber)
            .enforce(RuleKind::MuonLeptonNumber)
            .enforce(RuleKind::TauLeptonNumber)
            .enforce(RuleKind::LeptonNumber)
            .enforce(RuleKind::IdenticalParticleSymmetry)
    }

    /// Enable a rule in place using permissive missing-input handling.
    pub fn enforce_mut(&mut self, rule: RuleKind) -> &mut Self {
        self.set_policy_mut(rule, RulePolicy::enforce())
    }

    /// Enable a rule in place and reject candidates with missing inputs.
    pub fn enforce_strict_mut(&mut self, rule: RuleKind) -> &mut Self {
        self.set_policy_mut(rule, RulePolicy::enforce_strict())
    }

    /// Assign an explicit policy to a rule in place.
    pub fn set_policy_mut(&mut self, rule: RuleKind, policy: RulePolicy) -> &mut Self {
        self.policies.insert(rule, policy);
        self
    }

    /// Ignore a rule in place and record the supplied reason.
    pub fn ignore_mut(&mut self, rule: RuleKind, reason: impl Into<String>) -> &mut Self {
        self.set_policy_mut(rule, RulePolicy::ignore(reason))
    }

    /// Ignore a rule in place without recording a reason.
    pub fn ignore_without_reason_mut(&mut self, rule: RuleKind) -> &mut Self {
        self.set_policy_mut(rule, RulePolicy::ignore_without_reason())
    }

    /// Make a rule diagnostic-only in place and record the supplied reason.
    pub fn diagnose_only_mut(&mut self, rule: RuleKind, reason: impl Into<String>) -> &mut Self {
        self.set_policy_mut(rule, RulePolicy::diagnose_only(reason))
    }

    /// Make a rule diagnostic-only in place without recording a reason.
    pub fn diagnose_only_without_reason_mut(&mut self, rule: RuleKind) -> &mut Self {
        self.set_policy_mut(rule, RulePolicy::diagnose_only_without_reason())
    }

    /// Remove a rule from this set in place.
    pub fn disable_mut(&mut self, rule: RuleKind) -> &mut Self {
        self.policies.remove(&rule);
        self
    }

    /// Change a rule's missing-input policy in place.
    ///
    /// The rule is enabled with [`RulePolicy::enforce`] if it was not already
    /// configured.
    pub fn with_unknown_policy_mut(&mut self, rule: RuleKind, unknown: UnknownPolicy) -> &mut Self {
        self.policies
            .entry(rule)
            .or_insert_with(RulePolicy::enforce)
            .unknown = unknown;
        self
    }

    /// Return a copy with a permissively enforced rule.
    pub fn enforce(mut self, rule: RuleKind) -> Self {
        self.enforce_mut(rule);
        self
    }

    /// Return a copy with a strictly enforced rule.
    pub fn enforce_strict(mut self, rule: RuleKind) -> Self {
        self.enforce_strict_mut(rule);
        self
    }

    /// Return a copy with an explicit policy assigned to a rule.
    pub fn set_policy(mut self, rule: RuleKind, policy: RulePolicy) -> Self {
        self.set_policy_mut(rule, policy);
        self
    }

    /// Return a copy which ignores a rule for the supplied reason.
    pub fn ignore(mut self, rule: RuleKind, reason: impl Into<String>) -> Self {
        self.ignore_mut(rule, reason);
        self
    }

    /// Return a copy which ignores a rule without recording a reason.
    pub fn ignore_without_reason(mut self, rule: RuleKind) -> Self {
        self.ignore_without_reason_mut(rule);
        self
    }

    /// Return a copy which evaluates a rule only for diagnostics.
    pub fn diagnose_only(mut self, rule: RuleKind, reason: impl Into<String>) -> Self {
        self.diagnose_only_mut(rule, reason);
        self
    }

    /// Return a copy which evaluates a rule only for diagnostics, without a reason.
    pub fn diagnose_only_without_reason(mut self, rule: RuleKind) -> Self {
        self.diagnose_only_without_reason_mut(rule);
        self
    }

    /// Return a copy with a rule removed.
    pub fn disable(mut self, rule: RuleKind) -> Self {
        self.disable_mut(rule);
        self
    }

    /// Return a copy with the selected missing-input policy.
    pub fn with_unknown_policy(mut self, rule: RuleKind, unknown: UnknownPolicy) -> Self {
        self.with_unknown_policy_mut(rule, unknown);
        self
    }

    /// Retrieve the configured policy for a rule.
    pub fn policy(&self, rule: RuleKind) -> Option<&RulePolicy> {
        self.policies.get(&rule)
    }

    /// Iterate over the configured rules in stable [`RuleKind`] order.
    pub fn enabled_rules(&self) -> impl Iterator<Item = RuleKind> + '_ {
        self.policies.keys().copied()
    }

    /// Return whether a two-body partial-wave candidate satisfies this rule set.
    pub fn check(
        &self,
        parent: &ParticleProperties,
        daughters: (&ParticleProperties, &ParticleProperties),
        l: L,
        s: S,
    ) -> bool {
        self.evaluate(parent, daughters, l, s).is_allowed()
    }

    /// Evaluate every configured rule and return a detailed report.
    pub fn evaluate(
        &self,
        parent: &ParticleProperties,
        daughters: (&ParticleProperties, &ParticleProperties),
        l: L,
        s: S,
    ) -> RuleReport {
        let input = evaluators::RuleInput {
            parent,
            daughters,
            l,
            s,
        };
        let checks = self
            .policies
            .iter()
            .map(|(&rule, policy)| apply_policy(rule, policy, evaluators::evaluate(rule, input)))
            .collect();
        RuleReport { checks }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum RawRuleOutcome {
    Pass {
        message: String,
    },
    Fail {
        message: String,
    },
    Unknown {
        missing: Vec<String>,
        message: String,
    },
}

impl RawRuleOutcome {
    fn pass(message: impl Into<String>) -> Self {
        Self::Pass {
            message: message.into(),
        }
    }

    fn fail(message: impl Into<String>) -> Self {
        Self::Fail {
            message: message.into(),
        }
    }

    fn unknown(missing: impl Into<Vec<String>>, message: impl Into<String>) -> Self {
        Self::Unknown {
            missing: missing.into(),
            message: message.into(),
        }
    }

    fn message(&self) -> String {
        match self {
            Self::Pass { message } => message.clone(),
            Self::Fail { message } => message.clone(),
            Self::Unknown { message, .. } => message.clone(),
        }
    }

    fn passed(&self) -> Option<bool> {
        match self {
            Self::Pass { .. } => Some(true),
            Self::Fail { .. } => Some(false),
            Self::Unknown { .. } => None,
        }
    }
}

fn missing(fields: &[&'static str]) -> Vec<String> {
    fields.iter().map(|field| (*field).to_string()).collect()
}

fn apply_policy(rule: RuleKind, policy: &RulePolicy, raw: RawRuleOutcome) -> RuleCheck {
    let outcome = match &policy.mode {
        RuleMode::Ignore { reason } => RuleOutcome::Ignored {
            reason: reason.clone(),
        },

        RuleMode::DiagnoseOnly { reason } => RuleOutcome::Diagnostic {
            passed: raw.passed(),
            reason: reason.clone(),
            message: raw.message(),
        },

        RuleMode::Enforce => match raw {
            RawRuleOutcome::Pass { message } => RuleOutcome::Pass { message },

            RawRuleOutcome::Fail { message } => RuleOutcome::Fail { message },

            RawRuleOutcome::Unknown { missing, message } => match policy.unknown {
                UnknownPolicy::Allow => RuleOutcome::UnknownAllowed { missing, message },
                UnknownPolicy::Warn => RuleOutcome::Warning { missing, message },
                UnknownPolicy::Reject => RuleOutcome::Fail {
                    message: format!("{message}; unknown inputs are rejected by policy"),
                },
            },
        },
    };

    RuleCheck { rule, outcome }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::{
        j, l, m,
        quantum::{
            AllowedPartialWave, Isospin, J, M, Parity, PartialWave, PartialWaveCandidate,
            SelectionRules, Statistics,
        },
    };

    fn labels(waves: &[AllowedPartialWave]) -> Vec<String> {
        waves.iter().map(|w| w.wave.label()).collect()
    }

    fn allowed_labels<'a>(waves: impl Iterator<Item = &'a AllowedPartialWave>) -> Vec<String> {
        waves.map(|w| w.wave.label()).collect()
    }

    fn candidate_labels<'a>(
        candidates: impl Iterator<Item = &'a PartialWaveCandidate>,
    ) -> Vec<String> {
        candidates.map(|candidate| candidate.wave.label()).collect()
    }

    fn outcome(report: &RuleReport, rule: RuleKind) -> &RuleOutcome {
        report
            .outcome(rule)
            .unwrap_or_else(|| panic!("missing outcome for {rule:?}; report was {report:#?}"))
    }

    fn assert_pass(report: &RuleReport, rule: RuleKind) {
        assert!(
            matches!(outcome(report, rule), RuleOutcome::Pass { .. }),
            "expected {rule:?} to pass; got {:#?}",
            outcome(report, rule)
        );
    }

    fn assert_fail(report: &RuleReport, rule: RuleKind) {
        assert!(
            matches!(outcome(report, rule), RuleOutcome::Fail { .. }),
            "expected {rule:?} to fail; got {:#?}",
            outcome(report, rule)
        );
    }

    fn assert_unknown_allowed(report: &RuleReport, rule: RuleKind, expected_missing: &[&str]) {
        match outcome(report, rule) {
            RuleOutcome::UnknownAllowed { missing, .. } => {
                for field in expected_missing {
                    assert!(
                        missing.iter().any(|missing| missing == field),
                        "expected missing field {field:?}; got {missing:?}"
                    );
                }
            }
            other => panic!("expected {rule:?} to be UnknownAllowed; got {other:#?}"),
        }
    }

    fn assert_warning(report: &RuleReport, rule: RuleKind, expected_missing: &[&str]) {
        match outcome(report, rule) {
            RuleOutcome::Warning { missing, .. } => {
                for field in expected_missing {
                    assert!(
                        missing.iter().any(|missing| missing == field),
                        "expected missing field {field:?}; got {missing:?}"
                    );
                }
            }
            other => panic!("expected {rule:?} to be Warning; got {other:#?}"),
        }
    }

    fn assert_ignored(report: &RuleReport, rule: RuleKind, expected_reason: Option<&str>) {
        match outcome(report, rule) {
            RuleOutcome::Ignored { reason } => {
                assert_eq!(reason.as_deref(), expected_reason);
            }
            other => panic!("expected {rule:?} to be Ignored; got {other:#?}"),
        }
    }

    fn assert_diagnostic(
        report: &RuleReport,
        rule: RuleKind,
        expected_passed: Option<bool>,
        expected_reason: Option<&str>,
    ) {
        match outcome(report, rule) {
            RuleOutcome::Diagnostic { passed, reason, .. } => {
                assert_eq!(*passed, expected_passed);
                assert_eq!(reason.as_deref(), expected_reason);
            }
            other => panic!("expected {rule:?} to be Diagnostic; got {other:#?}"),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn add_additives(
        particle: ParticleProperties,
        charge: i32,
        strangeness: i32,
        charm: i32,
        bottomness: i32,
        topness: i32,
        baryon_number: i32,
        electron_lepton_number: i32,
        muon_lepton_number: i32,
        tau_lepton_number: i32,
    ) -> ParticleProperties {
        particle
            .with_charge(charge)
            .with_strangeness(strangeness)
            .unwrap()
            .with_charm(charm)
            .unwrap()
            .with_bottomness(bottomness)
            .unwrap()
            .with_topness(topness)
            .unwrap()
            .with_baryon_number(baryon_number)
            .unwrap()
            .with_electron_lepton_number(electron_lepton_number)
            .unwrap()
            .with_muon_lepton_number(muon_lepton_number)
            .unwrap()
            .with_tau_lepton_number(tau_lepton_number)
            .unwrap()
    }

    fn pion_like(name: &str, anti_name: &str, charge: i32, i3: i32) -> ParticleProperties {
        ParticleProperties::meson()
            .with_zero_flavor()
            .with_name(name)
            .with_species_names(name, anti_name)
            .unwrap()
            .with_spin(j!(0))
            .with_parity(Parity::Negative)
            .with_charge(charge)
            .with_isospin(Isospin::new(j!(1), Some(M::int(i3))).unwrap())
            .with_g_parity(Parity::Negative)
            .with_statistics(Statistics::Boson)
            .unwrap()
    }

    fn rho_like() -> ParticleProperties {
        ParticleProperties::meson()
            .with_zero_flavor()
            .with_name("rho0")
            .with_self_conjugate_species("rho0")
            .unwrap()
            .with_spin(j!(1))
            .with_parity(Parity::Negative)
            .with_c_parity(Parity::Negative)
            .unwrap()
            .with_charge(0)
            .with_isospin(Isospin::new(j!(1), Some(m!(0))).unwrap())
            .with_g_parity(Parity::Positive)
            .with_statistics(Statistics::Boson)
            .unwrap()
    }

    fn exotic_one_minus_plus() -> ParticleProperties {
        ParticleProperties::meson()
            .with_zero_flavor()
            .with_name("pi1_exotic")
            .with_self_conjugate_species("pi1_exotic")
            .unwrap()
            .with_spin(j!(1))
            .with_parity(Parity::Negative)
            .with_c_parity(Parity::Positive)
            .unwrap()
            .with_charge(0)
            .with_isospin(Isospin::new(j!(1), Some(m!(0))).unwrap())
            .with_statistics(Statistics::Boson)
            .unwrap()
    }

    fn identical_boson(spin: J, species: &str) -> ParticleProperties {
        ParticleProperties::unknown()
            .with_spin(spin)
            .with_species(species)
            .unwrap()
            .with_statistics(Statistics::Boson)
            .unwrap()
    }

    fn identical_fermion(spin: J, species: &str) -> ParticleProperties {
        ParticleProperties::unknown()
            .with_spin(spin)
            .with_species(species)
            .unwrap()
            .with_statistics(Statistics::Fermion)
            .unwrap()
    }

    const ALL_RULES: [RuleKind; 17] = [
        RuleKind::Parity,
        RuleKind::Isospin,
        RuleKind::IsospinProjection,
        RuleKind::CParity,
        RuleKind::GParity,
        RuleKind::Charge,
        RuleKind::Strangeness,
        RuleKind::Charm,
        RuleKind::Bottomness,
        RuleKind::Topness,
        RuleKind::BaryonNumber,
        RuleKind::ElectronLeptonNumber,
        RuleKind::MuonLeptonNumber,
        RuleKind::TauLeptonNumber,
        RuleKind::LeptonNumber,
        RuleKind::IdenticalParticleSymmetry,
        RuleKind::ConventionalMesonJpc,
    ];

    #[test]
    fn rule_registry_is_complete_unique_and_named() {
        let registered: Vec<_> = evaluators::registered_rules().collect();
        assert_eq!(
            registered.iter().map(|(rule, _)| *rule).collect::<Vec<_>>(),
            ALL_RULES,
        );
        assert!(registered.iter().all(|(_, name)| !name.is_empty()));
        assert_eq!(
            registered
                .iter()
                .map(|(rule, _)| *rule)
                .collect::<BTreeSet<_>>()
                .len(),
            ALL_RULES.len(),
        );
    }

    #[test]
    fn every_rule_outcome_obeys_every_policy_state() {
        let raw_outcomes = [
            RawRuleOutcome::pass("passed"),
            RawRuleOutcome::fail("failed"),
            RawRuleOutcome::unknown(vec!["missing.field".to_string()], "unknown"),
        ];
        let modes = [
            RuleMode::Enforce,
            RuleMode::Ignore { reason: None },
            RuleMode::DiagnoseOnly { reason: None },
        ];
        let unknown_policies = [
            UnknownPolicy::Allow,
            UnknownPolicy::Warn,
            UnknownPolicy::Reject,
        ];

        for rule in ALL_RULES {
            for mode in &modes {
                for unknown in unknown_policies {
                    for raw in &raw_outcomes {
                        let check = apply_policy(
                            rule,
                            &RulePolicy {
                                mode: mode.clone(),
                                unknown,
                            },
                            raw.clone(),
                        );
                        assert_eq!(check.rule, rule);
                        match (mode, raw, unknown, check.outcome) {
                            (RuleMode::Ignore { .. }, _, _, RuleOutcome::Ignored { .. })
                            | (
                                RuleMode::DiagnoseOnly { .. },
                                _,
                                _,
                                RuleOutcome::Diagnostic { .. },
                            )
                            | (
                                RuleMode::Enforce,
                                RawRuleOutcome::Pass { .. },
                                _,
                                RuleOutcome::Pass { .. },
                            )
                            | (
                                RuleMode::Enforce,
                                RawRuleOutcome::Fail { .. },
                                _,
                                RuleOutcome::Fail { .. },
                            )
                            | (
                                RuleMode::Enforce,
                                RawRuleOutcome::Unknown { .. },
                                UnknownPolicy::Allow,
                                RuleOutcome::UnknownAllowed { .. },
                            )
                            | (
                                RuleMode::Enforce,
                                RawRuleOutcome::Unknown { .. },
                                UnknownPolicy::Warn,
                                RuleOutcome::Warning { .. },
                            )
                            | (
                                RuleMode::Enforce,
                                RawRuleOutcome::Unknown { .. },
                                UnknownPolicy::Reject,
                                RuleOutcome::Fail { .. },
                            ) => {}
                            combination => panic!("unexpected policy result: {combination:?}"),
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn rule_presets_preserve_exact_membership_and_order() {
        let assert_rules = |actual: RuleSet, expected: &[RuleKind]| {
            assert_eq!(actual.enabled_rules().collect::<Vec<_>>(), expected);
        };
        assert_rules(
            RuleSet::strong(),
            &[
                RuleKind::Parity,
                RuleKind::Isospin,
                RuleKind::IsospinProjection,
                RuleKind::Charge,
                RuleKind::Strangeness,
                RuleKind::Charm,
                RuleKind::Bottomness,
                RuleKind::Topness,
                RuleKind::BaryonNumber,
                RuleKind::IdenticalParticleSymmetry,
            ],
        );
        assert_rules(
            RuleSet::electromagnetic(),
            &[
                RuleKind::Parity,
                RuleKind::IsospinProjection,
                RuleKind::Charge,
                RuleKind::Strangeness,
                RuleKind::Charm,
                RuleKind::Bottomness,
                RuleKind::Topness,
                RuleKind::BaryonNumber,
                RuleKind::IdenticalParticleSymmetry,
            ],
        );
        assert_rules(
            RuleSet::weak(),
            &[
                RuleKind::Charge,
                RuleKind::BaryonNumber,
                RuleKind::ElectronLeptonNumber,
                RuleKind::MuonLeptonNumber,
                RuleKind::TauLeptonNumber,
                RuleKind::LeptonNumber,
                RuleKind::IdenticalParticleSymmetry,
            ],
        );
    }

    #[test]
    fn representative_rule_policy_and_report_serde_stay_stable() {
        let policy = RulePolicy {
            mode: RuleMode::DiagnoseOnly {
                reason: Some("classification only".to_string()),
            },
            unknown: UnknownPolicy::Warn,
        };
        let policy_json = serde_json::to_string(&policy).unwrap();
        assert_eq!(
            policy_json,
            r#"{"mode":{"DiagnoseOnly":{"reason":"classification only"}},"unknown":"Warn"}"#,
        );
        assert_eq!(
            serde_json::from_str::<RulePolicy>(&policy_json).unwrap(),
            policy
        );

        let report = RuleReport {
            checks: vec![RuleCheck {
                rule: RuleKind::Charge,
                outcome: RuleOutcome::Warning {
                    missing: vec!["daughter_b.charge".to_string()],
                    message: "charge cannot be checked".to_string(),
                },
            }],
        };
        let report_json = serde_json::to_string(&report).unwrap();
        assert_eq!(
            report_json,
            r#"{"checks":[{"rule":"Charge","outcome":{"Warning":{"missing":["daughter_b.charge"],"message":"charge cannot be checked"}}}]}"#,
        );
        assert_eq!(
            serde_json::from_str::<RuleReport>(&report_json).unwrap(),
            report
        );
    }

    #[test]
    fn rule_set_constructors_build_expected_default_policies() {
        let angular = RuleSet::angular();
        assert_eq!(angular.enabled_rules().count(), 0);

        let strong = RuleSet::strong();
        for rule in [
            RuleKind::Parity,
            RuleKind::Isospin,
            RuleKind::IsospinProjection,
            RuleKind::Charge,
            RuleKind::Strangeness,
            RuleKind::Charm,
            RuleKind::Bottomness,
            RuleKind::Topness,
            RuleKind::BaryonNumber,
            RuleKind::IdenticalParticleSymmetry,
        ] {
            assert!(
                matches!(strong.policy(rule).unwrap().mode, RuleMode::Enforce),
                "strong rules should enforce {rule:?}"
            );
        }
        assert!(strong.policy(RuleKind::CParity).is_none());
        assert!(strong.policy(RuleKind::GParity).is_none());
        assert!(strong.policy(RuleKind::ConventionalMesonJpc).is_none());

        let electromagnetic = RuleSet::electromagnetic();
        assert!(electromagnetic.policy(RuleKind::Parity).is_some());
        assert!(electromagnetic.policy(RuleKind::Charge).is_some());
        assert!(
            electromagnetic
                .policy(RuleKind::IsospinProjection)
                .is_some()
        );
        assert!(electromagnetic.policy(RuleKind::Isospin).is_none());

        let weak = RuleSet::weak();
        assert!(weak.policy(RuleKind::Charge).is_some());
        assert!(weak.policy(RuleKind::BaryonNumber).is_some());
        assert!(weak.policy(RuleKind::ElectronLeptonNumber).is_some());
        assert!(weak.policy(RuleKind::MuonLeptonNumber).is_some());
        assert!(weak.policy(RuleKind::TauLeptonNumber).is_some());
        assert!(weak.policy(RuleKind::LeptonNumber).is_some());
        assert!(weak.policy(RuleKind::Parity).is_none());
        assert!(weak.policy(RuleKind::Strangeness).is_none());
    }

    #[test]
    fn rule_set_builder_and_mut_methods_configure_the_same_policies() {
        let built = RuleSet::angular()
            .enforce(RuleKind::Parity)
            .enforce_strict(RuleKind::Charge)
            .set_policy(
                RuleKind::Strangeness,
                RulePolicy::enforce_with_unknown_policy(UnknownPolicy::Warn),
            )
            .ignore(RuleKind::Isospin, "intentional isospin violation")
            .ignore_without_reason(RuleKind::GParity)
            .diagnose_only(
                RuleKind::ConventionalMesonJpc,
                "classify exotics without rejecting them",
            )
            .diagnose_only_without_reason(RuleKind::CParity)
            .with_unknown_policy(RuleKind::Bottomness, UnknownPolicy::Reject)
            .disable(RuleKind::CParity);

        let mut mutated = RuleSet::angular();
        mutated
            .enforce_mut(RuleKind::Parity)
            .enforce_strict_mut(RuleKind::Charge)
            .set_policy_mut(
                RuleKind::Strangeness,
                RulePolicy::enforce_with_unknown_policy(UnknownPolicy::Warn),
            )
            .ignore_mut(RuleKind::Isospin, "intentional isospin violation")
            .ignore_without_reason_mut(RuleKind::GParity)
            .diagnose_only_mut(
                RuleKind::ConventionalMesonJpc,
                "classify exotics without rejecting them",
            )
            .diagnose_only_without_reason_mut(RuleKind::CParity)
            .with_unknown_policy_mut(RuleKind::Bottomness, UnknownPolicy::Reject)
            .disable_mut(RuleKind::CParity);

        assert_eq!(built, mutated);

        assert!(matches!(
            built.policy(RuleKind::Parity).unwrap().mode,
            RuleMode::Enforce
        ));
        assert_eq!(
            built.policy(RuleKind::Charge).unwrap().unknown,
            UnknownPolicy::Reject
        );
        assert_eq!(
            built.policy(RuleKind::Strangeness).unwrap().unknown,
            UnknownPolicy::Warn
        );
        assert!(matches!(
            built.policy(RuleKind::Isospin).unwrap().mode,
            RuleMode::Ignore { reason: Some(_) }
        ));
        assert!(matches!(
            built.policy(RuleKind::GParity).unwrap().mode,
            RuleMode::Ignore { reason: None }
        ));
        assert!(matches!(
            built.policy(RuleKind::ConventionalMesonJpc).unwrap().mode,
            RuleMode::DiagnoseOnly { reason: Some(_) }
        ));
        assert_eq!(
            built.policy(RuleKind::Bottomness).unwrap().unknown,
            UnknownPolicy::Reject
        );
        assert!(built.policy(RuleKind::CParity).is_none());
    }

    #[test]
    fn policy_application_distinguishes_unknown_allowed_warning_reject_ignore_and_diagnostic() {
        let parent = ParticleProperties::unknown().with_spin(j!(0));
        let a = ParticleProperties::unknown().with_spin(j!(0));
        let b = ParticleProperties::unknown().with_spin(j!(0));

        let rules = RuleSet::angular()
            .enforce(RuleKind::Parity)
            .set_policy(
                RuleKind::Charge,
                RulePolicy::enforce_with_unknown_policy(UnknownPolicy::Warn),
            )
            .enforce_strict(RuleKind::Strangeness)
            .ignore(RuleKind::Isospin, "not relevant for this model")
            .diagnose_only(
                RuleKind::ConventionalMesonJpc,
                "only classify the parent assignment",
            );

        let report = rules.evaluate(&parent, (&a, &b), l!(0), j!(0));

        assert!(!report.is_allowed());
        assert_eq!(report.len(), 5);
        assert!(report.has_failures());
        assert!(report.has_unknowns());
        assert!(report.has_ignored());
        assert_eq!(report.failures().count(), 1);
        assert_eq!(report.warnings().count(), 1);
        assert_eq!(report.unknowns().count(), 1);
        assert_eq!(report.ignored().count(), 1);

        assert_unknown_allowed(&report, RuleKind::Parity, &["parent.parity"]);
        assert_warning(
            &report,
            RuleKind::Charge,
            &["parent.charge", "daughter_a.charge", "daughter_b.charge"],
        );
        assert_fail(&report, RuleKind::Strangeness);
        assert_ignored(
            &report,
            RuleKind::Isospin,
            Some("not relevant for this model"),
        );
        assert_diagnostic(
            &report,
            RuleKind::ConventionalMesonJpc,
            None,
            Some("only classify the parent assignment"),
        );
    }

    #[test]
    fn angular_momentum_helpers_partial_wave_validation_and_inference_work_together() {
        assert_eq!(
            SelectionRules::coupled_spins(j!(1 / 2), j!(1 / 2)),
            vec![j!(0), j!(1)]
        );
        assert_eq!(
            SelectionRules::coupled_spins(j!(1 / 2), j!(1)),
            vec![j!(1 / 2), j!(3 / 2)]
        );
        assert_eq!(
            SelectionRules::coupled_spins(j!(1), j!(1)),
            vec![j!(0), j!(1), j!(2)]
        );

        let wave = PartialWave::new(j!(1), l!(1), j!(0)).unwrap();
        assert_eq!(wave.label(), "1P1");
        assert_eq!(wave.to_string(), "1P1");

        assert!(PartialWave::new(j!(1), l!(0), j!(0)).is_err());

        let pi_plus = pion_like("pi+", "pi-", 1, 1);
        let pi_minus = pion_like("pi-", "pi+", -1, -1);
        let allowed = AllowedPartialWave::new(wave, (&pi_plus, &pi_minus));

        assert_eq!(allowed.parity, Some(Parity::Negative));
        assert_eq!(allowed.c_parity, Some(Parity::Negative));

        let non_c_pair = AllowedPartialWave::new(
            PartialWave::new(j!(0), l!(0), j!(0)).unwrap(),
            (&pi_plus, &pion_like("pi0", "pi0", 0, 0)),
        );
        assert_eq!(non_c_pair.parity, Some(Parity::Positive));
        assert_eq!(non_c_pair.c_parity, None);
    }

    #[test]
    fn complete_strong_plus_c_and_g_rules_pass_for_rho_like_to_charged_pions() {
        let parent = rho_like();
        let pi_plus = pion_like("pi+", "pi-", 1, 1);
        let pi_minus = pion_like("pi-", "pi+", -1, -1);

        let rules = RuleSet::strong()
            .enforce(RuleKind::CParity)
            .enforce(RuleKind::GParity)
            .enforce(RuleKind::ElectronLeptonNumber)
            .enforce(RuleKind::MuonLeptonNumber)
            .enforce(RuleKind::TauLeptonNumber)
            .enforce(RuleKind::LeptonNumber);

        let report = rules.evaluate(&parent, (&pi_plus, &pi_minus), l!(1), j!(0));

        assert!(report.is_allowed());
        assert_eq!(report.len(), rules.enabled_rules().count());
        assert!(
            report
                .checks
                .iter()
                .all(|check| { matches!(check.outcome, RuleOutcome::Pass { .. }) })
        );

        for rule in rules.enabled_rules() {
            assert_pass(&report, rule);
        }
    }

    #[test]
    fn nontrivial_quantum_number_rules_report_failures() {
        let parent = ParticleProperties::meson()
            .with_zero_flavor()
            .with_name("bad_parent")
            .with_self_conjugate_species("bad_parent")
            .unwrap()
            .with_spin(j!(1))
            .with_parity(Parity::Positive)
            .with_c_parity(Parity::Positive)
            .unwrap()
            .with_charge(0)
            .with_isospin(Isospin::new(j!(3), Some(m!(1))).unwrap())
            .with_g_parity(Parity::Negative)
            .with_statistics(Statistics::Boson)
            .unwrap();

        let pi_plus = pion_like("pi+", "pi-", 1, 1);
        let pi_minus = pion_like("pi-", "pi+", -1, -1);

        let rules = RuleSet::angular()
            .enforce(RuleKind::Parity)
            .enforce(RuleKind::Isospin)
            .enforce(RuleKind::IsospinProjection)
            .enforce(RuleKind::CParity)
            .enforce(RuleKind::GParity);

        let report = rules.evaluate(&parent, (&pi_plus, &pi_minus), l!(1), j!(0));

        assert!(!report.is_allowed());
        assert_eq!(report.failures().count(), 5);

        assert_fail(&report, RuleKind::Parity);
        assert_fail(&report, RuleKind::Isospin);
        assert_fail(&report, RuleKind::IsospinProjection);
        assert_fail(&report, RuleKind::CParity);
        assert_fail(&report, RuleKind::GParity);
    }

    #[test]
    fn additive_and_lepton_rules_report_passes_failures_and_missing_fields() {
        let parent = add_additives(
            ParticleProperties::unknown(),
            1, // charge violation against 0 + 0
            1, // strangeness violation
            1, // charm violation
            1, // bottomness violation
            1, // topness violation
            1, // baryon-number violation
            1, // electron-family lepton-number violation
            0,
            0,
        );

        let daughter_a = add_additives(ParticleProperties::unknown(), 0, 0, 0, 0, 0, 0, 0, 1, 0);
        let daughter_b = add_additives(ParticleProperties::unknown(), 0, 0, 0, 0, 0, 0, 0, 0, 0);

        let rules = RuleSet::angular()
            .enforce(RuleKind::Charge)
            .enforce(RuleKind::Strangeness)
            .enforce(RuleKind::Charm)
            .enforce(RuleKind::Bottomness)
            .enforce(RuleKind::Topness)
            .enforce(RuleKind::BaryonNumber)
            .enforce(RuleKind::ElectronLeptonNumber)
            .enforce(RuleKind::MuonLeptonNumber)
            .enforce(RuleKind::TauLeptonNumber)
            .enforce(RuleKind::LeptonNumber);

        let report = rules.evaluate(&parent, (&daughter_a, &daughter_b), l!(0), j!(0));

        assert!(!report.is_allowed());

        for rule in [
            RuleKind::Charge,
            RuleKind::Strangeness,
            RuleKind::Charm,
            RuleKind::Bottomness,
            RuleKind::Topness,
            RuleKind::BaryonNumber,
            RuleKind::ElectronLeptonNumber,
            RuleKind::MuonLeptonNumber,
        ] {
            assert_fail(&report, rule);
        }

        assert_pass(&report, RuleKind::TauLeptonNumber);
        assert_pass(&report, RuleKind::LeptonNumber);

        let parent_missing = ParticleProperties::unknown().with_charge(0);
        let a_missing = ParticleProperties::unknown().with_charge(0);
        let b_missing = ParticleProperties::unknown();

        let unknown_report = RuleSet::angular()
            .enforce(RuleKind::Charge)
            .enforce(RuleKind::LeptonNumber)
            .evaluate(&parent_missing, (&a_missing, &b_missing), l!(0), j!(0));

        assert!(unknown_report.is_allowed());
        assert_unknown_allowed(&unknown_report, RuleKind::Charge, &["daughter_b.charge"]);
        assert_unknown_allowed(
            &unknown_report,
            RuleKind::LeptonNumber,
            &[
                "parent.electron_lepton_number",
                "daughter_a.electron_lepton_number",
                "daughter_b.electron_lepton_number",
            ],
        );
    }

    #[test]
    fn identical_particle_symmetry_handles_bosons_and_fermions_with_spin_dependence() {
        let scalar_a = identical_boson(j!(0), "scalar");
        let scalar_b = identical_boson(j!(0), "scalar");

        let vector_a = identical_boson(j!(1), "vector");
        let vector_b = identical_boson(j!(1), "vector");

        let fermion_a = identical_fermion(j!(1 / 2), "fermion");
        let fermion_b = identical_fermion(j!(1 / 2), "fermion");

        let rules = RuleSet::angular().enforce(RuleKind::IdenticalParticleSymmetry);

        let scalar_even_l = rules.evaluate(
            &ParticleProperties::unknown(),
            (&scalar_a, &scalar_b),
            l!(0),
            j!(0),
        );
        assert_pass(&scalar_even_l, RuleKind::IdenticalParticleSymmetry);

        let scalar_odd_l = rules.evaluate(
            &ParticleProperties::unknown(),
            (&scalar_a, &scalar_b),
            l!(1),
            j!(0),
        );
        assert_fail(&scalar_odd_l, RuleKind::IdenticalParticleSymmetry);

        let vector_s0 = rules.evaluate(
            &ParticleProperties::unknown(),
            (&vector_a, &vector_b),
            l!(0),
            j!(0),
        );
        assert_pass(&vector_s0, RuleKind::IdenticalParticleSymmetry);

        let vector_s1 = rules.evaluate(
            &ParticleProperties::unknown(),
            (&vector_a, &vector_b),
            l!(0),
            j!(1),
        );
        assert_fail(&vector_s1, RuleKind::IdenticalParticleSymmetry);

        let fermion_s0 = rules.evaluate(
            &ParticleProperties::unknown(),
            (&fermion_a, &fermion_b),
            l!(0),
            j!(0),
        );
        assert_pass(&fermion_s0, RuleKind::IdenticalParticleSymmetry);

        let fermion_s1 = rules.evaluate(
            &ParticleProperties::unknown(),
            (&fermion_a, &fermion_b),
            l!(0),
            j!(1),
        );
        assert_fail(&fermion_s1, RuleKind::IdenticalParticleSymmetry);

        let different_species = rules.evaluate(
            &ParticleProperties::unknown(),
            (&identical_boson(j!(0), "a"), &identical_boson(j!(0), "b")),
            l!(1),
            j!(0),
        );
        assert_pass(&different_species, RuleKind::IdenticalParticleSymmetry);
    }

    #[test]
    fn c_parity_rule_distinguishes_inferred_c_from_non_inferable_final_states() {
        let parent = rho_like();
        let pi_plus = pion_like("pi+", "pi-", 1, 1);
        let pi_minus = pion_like("pi-", "pi+", -1, -1);
        let pi_zero = pion_like("pi0", "pi0", 0, 0);

        let rules = RuleSet::angular().enforce(RuleKind::CParity);

        let p_wave_report = rules.evaluate(&parent, (&pi_plus, &pi_minus), l!(1), j!(0));
        assert_pass(&p_wave_report, RuleKind::CParity);

        let s_wave_report = rules.evaluate(&parent, (&pi_plus, &pi_minus), l!(0), j!(0));
        assert_fail(&s_wave_report, RuleKind::CParity);

        let unknown_report = rules.evaluate(&parent, (&pi_plus, &pi_zero), l!(1), j!(0));
        assert!(unknown_report.is_allowed());
        assert_unknown_allowed(
            &unknown_report,
            RuleKind::CParity,
            &[
                "daughter_a.species",
                "daughter_a.antiparticle_species",
                "daughter_b.species",
                "daughter_b.antiparticle_species",
            ],
        );
    }

    #[test]
    fn conventional_meson_jpc_can_be_enforced_or_used_as_non_rejecting_diagnostic() {
        let conventional = rho_like();
        let exotic = exotic_one_minus_plus();

        let diagnostic_rules = RuleSet::angular().diagnose_only(
            RuleKind::ConventionalMesonJpc,
            "flag exotic JPC without rejecting hybrid candidates",
        );

        let conventional_report =
            diagnostic_rules.evaluate(&conventional, (&conventional, &conventional), l!(0), j!(0));
        assert!(conventional_report.is_allowed());
        assert_diagnostic(
            &conventional_report,
            RuleKind::ConventionalMesonJpc,
            Some(true),
            Some("flag exotic JPC without rejecting hybrid candidates"),
        );

        let exotic_report = diagnostic_rules.evaluate(&exotic, (&exotic, &exotic), l!(0), j!(0));
        assert!(exotic_report.is_allowed());
        assert_diagnostic(
            &exotic_report,
            RuleKind::ConventionalMesonJpc,
            Some(false),
            Some("flag exotic JPC without rejecting hybrid candidates"),
        );

        let enforced_report = RuleSet::angular()
            .enforce(RuleKind::ConventionalMesonJpc)
            .evaluate(&exotic, (&exotic, &exotic), l!(0), j!(0));
        assert!(!enforced_report.is_allowed());
        assert_fail(&enforced_report, RuleKind::ConventionalMesonJpc);

        let unknown_report = RuleSet::angular()
            .diagnose_only_without_reason(RuleKind::ConventionalMesonJpc)
            .evaluate(
                &ParticleProperties::unknown(),
                (
                    &ParticleProperties::unknown(),
                    &ParticleProperties::unknown(),
                ),
                l!(0),
                j!(0),
            );
        assert_diagnostic(&unknown_report, RuleKind::ConventionalMesonJpc, None, None);
    }

    #[test]
    fn selection_rules_scan_partial_waves_keeps_rejected_candidates_for_diagnostics() {
        let parent = ParticleProperties::jp(j!(1), Parity::Positive);
        let a = ParticleProperties::jp(j!(1 / 2), Parity::Positive);
        let b = ParticleProperties::jp(j!(1 / 2), Parity::Negative);

        let angular_scan = SelectionRules::angular(l!(2)).scan_partial_waves(&parent, (&a, &b));

        assert!(angular_scan.missing_inputs.is_empty());
        assert_eq!(
            candidate_labels(angular_scan.candidates.iter()),
            vec!["1P1", "3S1", "3P1", "3D1"]
        );
        assert_eq!(
            allowed_labels(angular_scan.allowed()),
            vec!["1P1", "3S1", "3P1", "3D1"]
        );
        assert_eq!(angular_scan.rejected().count(), 0);

        let parity_rules = SelectionRules::new(RuleSet::angular().enforce(RuleKind::Parity), l!(2));
        let parity_scan = parity_rules.scan_partial_waves(&parent, (&a, &b));

        assert_eq!(
            candidate_labels(parity_scan.candidates.iter()),
            vec!["1P1", "3S1", "3P1", "3D1"]
        );
        assert_eq!(allowed_labels(parity_scan.allowed()), vec!["1P1", "3P1"]);
        assert_eq!(candidate_labels(parity_scan.rejected()), vec!["3S1", "3D1"]);

        let allowed = parity_rules.allowed_partial_waves(&parent, (&a, &b));
        assert_eq!(labels(&allowed), vec!["1P1", "3P1"]);
    }

    #[test]
    fn selection_rules_report_missing_spin_inputs_and_default_to_strong_l6() {
        assert_eq!(SelectionRules::default(), SelectionRules::strong(l!(6)));
        assert_eq!(
            SelectionRules::electromagnetic(l!(2)),
            SelectionRules::new(RuleSet::electromagnetic(), l!(2))
        );
        assert_eq!(
            SelectionRules::weak(l!(3)),
            SelectionRules::new(RuleSet::weak(), l!(3))
        );

        let parent_missing = ParticleProperties::unknown();
        let a = ParticleProperties::jp(j!(0), Parity::Negative);
        let b = ParticleProperties::jp(j!(0), Parity::Negative);

        let scan = SelectionRules::default().scan_partial_waves(&parent_missing, (&a, &b));

        assert!(scan.candidates.is_empty());
        assert_eq!(scan.missing_inputs, vec!["parent.spin"]);

        let parent = ParticleProperties::jp(j!(0), Parity::Positive);
        let a_missing = ParticleProperties::unknown();

        let scan = SelectionRules::default().scan_partial_waves(&parent, (&a_missing, &b));

        assert!(scan.candidates.is_empty());
        assert_eq!(scan.missing_inputs, vec!["daughter_a.spin"]);

        let b_missing = ParticleProperties::unknown();

        let scan = SelectionRules::default().scan_partial_waves(&parent, (&a, &b_missing));

        assert!(scan.candidates.is_empty());
        assert_eq!(scan.missing_inputs, vec!["daughter_b.spin"]);
    }

    #[test]
    fn strong_rules_find_delta_like_to_nucleon_pion_p_wave() {
        let parent = add_additives(
            ParticleProperties::jp(j!(3 / 2), Parity::Positive),
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
        );

        let nucleon = add_additives(
            ParticleProperties::jp(j!(1 / 2), Parity::Positive),
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
        );

        let pion = add_additives(
            ParticleProperties::jp(j!(0), Parity::Negative),
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        );

        let rules = SelectionRules::new(
            RuleSet::angular()
                .enforce(RuleKind::Parity)
                .enforce(RuleKind::Charge)
                .enforce(RuleKind::BaryonNumber),
            l!(4),
        );

        let waves = rules.allowed_partial_waves(&parent, (&nucleon, &pion));

        assert_eq!(labels(&waves), vec!["2P3/2"]);
        assert_eq!(waves[0].parity, Some(Parity::Positive));
        assert_eq!(waves[0].c_parity, None);
    }
}
