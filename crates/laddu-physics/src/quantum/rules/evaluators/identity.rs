use crate::quantum::Statistics;

use super::RuleInput;
use crate::quantum::rules::{RawRuleOutcome, missing};

pub(super) fn identical_particle_symmetry(input: RuleInput<'_>) -> RawRuleOutcome {
    let Some(species_a) = input.daughters.0.species.as_ref() else {
        return RawRuleOutcome::unknown(
            missing(&["daughter_a.species"]),
            "first daughter species is unknown",
        );
    };
    let Some(species_b) = input.daughters.1.species.as_ref() else {
        return RawRuleOutcome::unknown(
            missing(&["daughter_b.species"]),
            "second daughter species is unknown",
        );
    };
    if species_a != species_b {
        return RawRuleOutcome::pass("daughters are not identical particles");
    }

    let Some(stats_a) = input.daughters.0.statistics else {
        return RawRuleOutcome::unknown(
            missing(&["daughter_a.statistics"]),
            "first daughter statistics are unknown",
        );
    };
    let Some(stats_b) = input.daughters.1.statistics else {
        return RawRuleOutcome::unknown(
            missing(&["daughter_b.statistics"]),
            "second daughter statistics are unknown",
        );
    };
    if stats_a != stats_b {
        return RawRuleOutcome::fail(
            "identical particles have inconsistent statistics assignments",
        );
    }

    let Some(ja) = input.daughters.0.spin else {
        return RawRuleOutcome::unknown(
            missing(&["daughter_a.spin"]),
            "first daughter spin is unknown",
        );
    };
    let Some(jb) = input.daughters.1.spin else {
        return RawRuleOutcome::unknown(
            missing(&["daughter_b.spin"]),
            "second daughter spin is unknown",
        );
    };
    if ja != jb {
        return RawRuleOutcome::fail("identical particles have inconsistent spin assignments");
    }
    if !input.s.doubled().is_multiple_of(2) {
        return RawRuleOutcome::fail(
            "two identical particles cannot couple to half-integer total spin",
        );
    }
    let coupled_spin = input.s.doubled() / 2;
    if coupled_spin > ja.doubled() {
        return RawRuleOutcome::fail(
            "coupled spin is incompatible with two identical daughter spins",
        );
    }

    let exchange_exponent = input.l.value() + ja.doubled() - coupled_spin;
    let exchange_is_symmetric = exchange_exponent.is_multiple_of(2);
    let allowed = match stats_a {
        Statistics::Boson => exchange_is_symmetric,
        Statistics::Fermion => !exchange_is_symmetric,
    };
    if allowed {
        RawRuleOutcome::pass("identical-particle exchange symmetry is satisfied")
    } else {
        RawRuleOutcome::fail("identical-particle exchange symmetry is violated")
    }
}
