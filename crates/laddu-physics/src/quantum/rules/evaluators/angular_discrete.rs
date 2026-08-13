use crate::quantum::partial_waves::{infer_c_parity, infer_parity};

use super::RuleInput;
use crate::quantum::rules::{RawRuleOutcome, missing};

pub(super) fn parity(input: RuleInput<'_>) -> RawRuleOutcome {
    let Some(parent) = input.parent.parity else {
        return RawRuleOutcome::unknown(missing(&["parent.parity"]), "parent parity is unknown");
    };
    let mut absent = Vec::new();
    if input.daughters.0.parity.is_none() {
        absent.push("daughter_a.parity".to_string());
    }
    if input.daughters.1.parity.is_none() {
        absent.push("daughter_b.parity".to_string());
    }
    if !absent.is_empty() {
        return RawRuleOutcome::unknown(
            absent,
            "final-state parity cannot be inferred because one or both daughter parities are unknown",
        );
    }

    let final_parity =
        infer_parity(input.daughters, input.l).expect("daughter parities were checked above");
    if parent == final_parity {
        RawRuleOutcome::pass(format!(
            "parity is conserved for L = {} with final parity {:?}",
            input.l.value(),
            final_parity,
        ))
    } else {
        RawRuleOutcome::fail(format!(
            "parity is not conserved for L = {}: parent parity is {:?}, final parity is {:?}",
            input.l.value(),
            parent,
            final_parity,
        ))
    }
}

pub(super) fn isospin(input: RuleInput<'_>) -> RawRuleOutcome {
    let Some(parent) = input.parent.isospin else {
        return RawRuleOutcome::unknown(missing(&["parent.isospin"]), "parent isospin is unknown");
    };
    let Some(a) = input.daughters.0.isospin else {
        return RawRuleOutcome::unknown(
            missing(&["daughter_a.isospin"]),
            "first daughter isospin is unknown",
        );
    };
    let Some(b) = input.daughters.1.isospin else {
        return RawRuleOutcome::unknown(
            missing(&["daughter_b.isospin"]),
            "second daughter isospin is unknown",
        );
    };

    if parent.isospin().can_couple_to(a.isospin(), b.isospin()) {
        RawRuleOutcome::pass("daughter isospins can couple to parent isospin")
    } else {
        RawRuleOutcome::fail("daughter isospins cannot couple to parent isospin")
    }
}

pub(super) fn isospin_projection(input: RuleInput<'_>) -> RawRuleOutcome {
    let Some(parent) = input.parent.isospin else {
        return RawRuleOutcome::unknown(missing(&["parent.isospin"]), "parent isospin is unknown");
    };
    let Some(a) = input.daughters.0.isospin else {
        return RawRuleOutcome::unknown(
            missing(&["daughter_a.isospin"]),
            "first daughter isospin is unknown",
        );
    };
    let Some(b) = input.daughters.1.isospin else {
        return RawRuleOutcome::unknown(
            missing(&["daughter_b.isospin"]),
            "second daughter isospin is unknown",
        );
    };
    let Some(parent_projection) = parent.projection else {
        return RawRuleOutcome::unknown(
            missing(&["parent.isospin.projection"]),
            "parent isospin projection is unknown",
        );
    };
    let Some(a_projection) = a.projection else {
        return RawRuleOutcome::unknown(
            missing(&["daughter_a.isospin.projection"]),
            "first daughter isospin projection is unknown",
        );
    };
    let Some(b_projection) = b.projection else {
        return RawRuleOutcome::unknown(
            missing(&["daughter_b.isospin.projection"]),
            "second daughter isospin projection is unknown",
        );
    };

    if parent_projection.doubled() == a_projection.doubled() + b_projection.doubled() {
        RawRuleOutcome::pass("isospin projection is conserved")
    } else {
        RawRuleOutcome::fail("isospin projection is not conserved")
    }
}

pub(super) fn c_parity(input: RuleInput<'_>) -> RawRuleOutcome {
    let Some(parent) = input.parent.c_parity else {
        return RawRuleOutcome::unknown(
            missing(&["parent.c_parity"]),
            "parent C-parity is unknown or not applicable",
        );
    };
    let Some(final_parity) = infer_c_parity(input.daughters, input.l, input.s) else {
        return RawRuleOutcome::unknown(
            missing(&[
                "daughter_a.species",
                "daughter_a.antiparticle_species",
                "daughter_b.species",
                "daughter_b.antiparticle_species",
            ]),
            "final-state C-parity cannot be inferred; this check currently assumes a C-eigenstate particle-antiparticle combination",
        );
    };

    if parent == final_parity {
        RawRuleOutcome::pass(format!(
            "C-parity is conserved with inferred C_final = {:?}; assumes a C-eigenstate particle-antiparticle combination",
            final_parity,
        ))
    } else {
        RawRuleOutcome::fail(format!(
            "C-parity is not conserved: parent C = {:?}, inferred final C = {:?}; assumes a C-eigenstate particle-antiparticle combination",
            parent, final_parity,
        ))
    }
}

pub(super) fn g_parity(input: RuleInput<'_>) -> RawRuleOutcome {
    let Some(parent) = input.parent.g_parity else {
        return RawRuleOutcome::unknown(
            missing(&["parent.g_parity"]),
            "parent G-parity is unknown or not applicable",
        );
    };
    let Some(a) = input.daughters.0.g_parity else {
        return RawRuleOutcome::unknown(
            missing(&["daughter_a.g_parity"]),
            "first daughter G-parity is unknown or not applicable",
        );
    };
    let Some(b) = input.daughters.1.g_parity else {
        return RawRuleOutcome::unknown(
            missing(&["daughter_b.g_parity"]),
            "second daughter G-parity is unknown or not applicable",
        );
    };
    let final_parity = a.value() * b.value();
    if parent.value() == final_parity {
        RawRuleOutcome::pass("G-parity product check passes")
    } else {
        RawRuleOutcome::fail(format!(
            "G-parity product check fails: parent G = {:?}, daughter product = {}",
            parent, final_parity,
        ))
    }
}
