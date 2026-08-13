use crate::quantum::L;

use super::RuleInput;
use crate::quantum::rules::{RawRuleOutcome, missing};

pub(super) fn conventional_meson_jpc(input: RuleInput<'_>) -> RawRuleOutcome {
    let Some(j) = input.parent.spin else {
        return RawRuleOutcome::unknown(missing(&["parent.spin"]), "parent spin is unknown");
    };
    let Some(p) = input.parent.parity else {
        return RawRuleOutcome::unknown(missing(&["parent.parity"]), "parent parity is unknown");
    };
    let Some(c) = input.parent.c_parity else {
        return RawRuleOutcome::unknown(
            missing(&["parent.c_parity"]),
            "parent C-parity is unknown or not applicable",
        );
    };
    if !j.doubled().is_multiple_of(2) {
        return RawRuleOutcome::fail(
            "half-integer J is not compatible with a conventional meson assignment",
        );
    }

    let target_j = j.doubled();
    for l in 0..=target_j / 2 + 1 {
        for s in [0u32, 1u32] {
            let l_doubled = 2 * l;
            let s_doubled = 2 * s;
            let angular_ok =
                target_j >= l_doubled.abs_diff(s_doubled) && target_j <= l_doubled + s_doubled;
            let parity_ok = p == L::int(l + 1).orbital_parity();
            let c_ok = c == L::int(l + s).orbital_parity();
            if angular_ok && parity_ok && c_ok {
                return RawRuleOutcome::pass(
                    "J^PC is compatible with a conventional q qbar meson assignment",
                );
            }
        }
    }

    RawRuleOutcome::fail(format!(
        "J^PC = {}{:?}{:?} is exotic for a conventional q qbar meson assignment",
        j, p, c,
    ))
}
