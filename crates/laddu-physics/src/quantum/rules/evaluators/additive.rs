use crate::quantum::ParticleProperties;

use super::RuleInput;
use crate::quantum::rules::RawRuleOutcome;

type Accessor = fn(&ParticleProperties) -> Option<i32>;

#[derive(Clone, Copy)]
pub(super) struct AdditiveField {
    field: &'static str,
    pub(super) display_name: &'static str,
    symbol: &'static str,
    accessor: Accessor,
}

impl AdditiveField {
    const fn new(
        field: &'static str,
        display_name: &'static str,
        symbol: &'static str,
        accessor: Accessor,
    ) -> Self {
        Self {
            field,
            display_name,
            symbol,
            accessor,
        }
    }

    pub(super) fn evaluate(self, input: RuleInput<'_>) -> RawRuleOutcome {
        let parent = (self.accessor)(input.parent);
        let a = (self.accessor)(input.daughters.0);
        let b = (self.accessor)(input.daughters.1);

        match (parent, a, b) {
            (Some(parent), Some(a), Some(b)) => {
                let final_value = a + b;
                if parent == final_value {
                    RawRuleOutcome::pass(format!("{} is conserved", self.display_name))
                } else {
                    RawRuleOutcome::fail(format!(
                        "{} is not conserved: parent {} = {}, final {} = {}",
                        self.display_name, self.symbol, parent, self.symbol, final_value,
                    ))
                }
            }
            _ => {
                let mut missing = Vec::new();
                if parent.is_none() {
                    missing.push(format!("parent.{}", self.field));
                }
                if a.is_none() {
                    missing.push(format!("daughter_a.{}", self.field));
                }
                if b.is_none() {
                    missing.push(format!("daughter_b.{}", self.field));
                }
                RawRuleOutcome::unknown(
                    missing,
                    format!(
                        "{} cannot be checked because required values are unknown",
                        self.display_name,
                    ),
                )
            }
        }
    }
}

macro_rules! additive_field {
    ($name:ident, $field:ident, $display:literal, $symbol:literal) => {
        pub(super) const $name: AdditiveField =
            AdditiveField::new(stringify!($field), $display, $symbol, |particle| {
                particle.$field
            });
    };
}

additive_field!(CHARGE, charge, "charge", "Q");
additive_field!(STRANGENESS, strangeness, "strangeness", "S");
additive_field!(CHARM, charm, "charm", "C");
additive_field!(BOTTOMNESS, bottomness, "bottomness", "B'");
additive_field!(TOPNESS, topness, "topness", "T");
additive_field!(BARYON_NUMBER, baryon_number, "baryon number", "B");
additive_field!(
    ELECTRON_LEPTON_NUMBER,
    electron_lepton_number,
    "electron-family lepton number",
    "L_e"
);
additive_field!(
    MUON_LEPTON_NUMBER,
    muon_lepton_number,
    "muon-family lepton number",
    "L_mu"
);
additive_field!(
    TAU_LEPTON_NUMBER,
    tau_lepton_number,
    "tau-family lepton number",
    "L_tau"
);

pub(super) fn total_lepton_number(input: RuleInput<'_>) -> RawRuleOutcome {
    let values = [
        (
            "parent.electron_lepton_number",
            input.parent.electron_lepton_number,
        ),
        ("parent.muon_lepton_number", input.parent.muon_lepton_number),
        ("parent.tau_lepton_number", input.parent.tau_lepton_number),
        (
            "daughter_a.electron_lepton_number",
            input.daughters.0.electron_lepton_number,
        ),
        (
            "daughter_a.muon_lepton_number",
            input.daughters.0.muon_lepton_number,
        ),
        (
            "daughter_a.tau_lepton_number",
            input.daughters.0.tau_lepton_number,
        ),
        (
            "daughter_b.electron_lepton_number",
            input.daughters.1.electron_lepton_number,
        ),
        (
            "daughter_b.muon_lepton_number",
            input.daughters.1.muon_lepton_number,
        ),
        (
            "daughter_b.tau_lepton_number",
            input.daughters.1.tau_lepton_number,
        ),
    ];
    let missing: Vec<String> = values
        .iter()
        .filter(|(_, value)| value.is_none())
        .map(|(name, _)| (*name).to_string())
        .collect();

    if !missing.is_empty() {
        return RawRuleOutcome::unknown(
            missing,
            "total lepton number cannot be checked because required values are unknown",
        );
    }

    let total = |particle: &ParticleProperties| {
        particle.electron_lepton_number.unwrap()
            + particle.muon_lepton_number.unwrap()
            + particle.tau_lepton_number.unwrap()
    };
    let parent = total(input.parent);
    let daughters = total(input.daughters.0) + total(input.daughters.1);

    if parent == daughters {
        RawRuleOutcome::pass("total lepton number is conserved")
    } else {
        RawRuleOutcome::fail(format!(
            "total lepton number is not conserved: parent L = {parent}, final L = {daughters}",
        ))
    }
}
