mod additive;
mod angular_discrete;
mod classification;
mod identity;

use crate::quantum::{L, ParticleProperties, S};

use super::{RawRuleOutcome, RuleKind};

type Evaluator = for<'a> fn(RuleInput<'a>) -> RawRuleOutcome;

#[derive(Clone, Copy)]
pub(super) struct RuleInput<'a> {
    pub(super) parent: &'a ParticleProperties,
    pub(super) daughters: (&'a ParticleProperties, &'a ParticleProperties),
    pub(super) l: L,
    pub(super) s: S,
}

#[derive(Clone, Copy)]
enum RuleEvaluator {
    Direct(Evaluator),
    Additive(additive::AdditiveField),
}

#[derive(Clone, Copy)]
struct RuleDescriptor {
    kind: RuleKind,
    display_name: &'static str,
    evaluator: RuleEvaluator,
}

impl RuleDescriptor {
    const fn direct(kind: RuleKind, display_name: &'static str, evaluator: Evaluator) -> Self {
        Self {
            kind,
            display_name,
            evaluator: RuleEvaluator::Direct(evaluator),
        }
    }

    const fn additive(kind: RuleKind, field: additive::AdditiveField) -> Self {
        Self {
            kind,
            display_name: field.display_name,
            evaluator: RuleEvaluator::Additive(field),
        }
    }

    fn evaluate(self, input: RuleInput<'_>) -> RawRuleOutcome {
        match self.evaluator {
            RuleEvaluator::Direct(evaluator) => evaluator(input),
            RuleEvaluator::Additive(field) => field.evaluate(input),
        }
    }
}

const RULE_REGISTRY: [RuleDescriptor; 17] = [
    RuleDescriptor::direct(RuleKind::Parity, "parity", angular_discrete::parity),
    RuleDescriptor::direct(RuleKind::Isospin, "isospin", angular_discrete::isospin),
    RuleDescriptor::direct(
        RuleKind::IsospinProjection,
        "isospin projection",
        angular_discrete::isospin_projection,
    ),
    RuleDescriptor::direct(RuleKind::CParity, "C-parity", angular_discrete::c_parity),
    RuleDescriptor::direct(RuleKind::GParity, "G-parity", angular_discrete::g_parity),
    RuleDescriptor::additive(RuleKind::Charge, additive::CHARGE),
    RuleDescriptor::additive(RuleKind::Strangeness, additive::STRANGENESS),
    RuleDescriptor::additive(RuleKind::Charm, additive::CHARM),
    RuleDescriptor::additive(RuleKind::Bottomness, additive::BOTTOMNESS),
    RuleDescriptor::additive(RuleKind::Topness, additive::TOPNESS),
    RuleDescriptor::additive(RuleKind::BaryonNumber, additive::BARYON_NUMBER),
    RuleDescriptor::additive(
        RuleKind::ElectronLeptonNumber,
        additive::ELECTRON_LEPTON_NUMBER,
    ),
    RuleDescriptor::additive(RuleKind::MuonLeptonNumber, additive::MUON_LEPTON_NUMBER),
    RuleDescriptor::additive(RuleKind::TauLeptonNumber, additive::TAU_LEPTON_NUMBER),
    RuleDescriptor::direct(
        RuleKind::LeptonNumber,
        "total lepton number",
        additive::total_lepton_number,
    ),
    RuleDescriptor::direct(
        RuleKind::IdenticalParticleSymmetry,
        "identical-particle symmetry",
        identity::identical_particle_symmetry,
    ),
    RuleDescriptor::direct(
        RuleKind::ConventionalMesonJpc,
        "conventional meson J^PC",
        classification::conventional_meson_jpc,
    ),
];

pub(super) fn evaluate(rule: RuleKind, input: RuleInput<'_>) -> RawRuleOutcome {
    let descriptor = RULE_REGISTRY
        .iter()
        .find(|descriptor| descriptor.kind == rule)
        .unwrap_or_else(|| panic!("selection rule {rule:?} is missing from the rule registry"));
    debug_assert!(!descriptor.display_name.is_empty());
    descriptor.evaluate(input)
}

#[cfg(test)]
pub(super) fn registered_rules() -> impl Iterator<Item = (RuleKind, &'static str)> {
    RULE_REGISTRY
        .iter()
        .map(|descriptor| (descriptor.kind, descriptor.display_name))
}
