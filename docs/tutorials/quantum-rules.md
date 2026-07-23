# Quantum-number selection rules

The Python selection-rule API exposes the two-body checks implemented in
`quantum/rules.rs`. It separates exact angular-momentum coupling from optional
interaction-specific conservation rules.

## Evaluate a proposed decay

```python
import laddu as ld

rho = ld.Particle(
    "rho0",
    species="rho0",
    self_conjugate=True,
    spin=1,
    parity="-",
    c_parity="-",
    charge=0,
    statistics=ld.Statistics.BOSON,
)
pi_plus = ld.Particle(
    "pi+",
    species="pi+",
    antiparticle_species="pi-",
    spin=0,
    parity="-",
    charge=1,
    statistics=ld.Statistics.BOSON,
)
pi_minus = ld.Particle(
    "pi-",
    species="pi-",
    antiparticle_species="pi+",
    spin=0,
    parity="-",
    charge=-1,
    statistics=ld.Statistics.BOSON,
)

rules = ld.RuleSet.strong().enforce("c_parity")
report = rules.evaluate(rho, pi_plus, pi_minus, ld.L(1), ld.S(0))

assert report.is_allowed
for check in report.checks:
    print(check.rule, check.outcome, check.message)
```

Standard constructors are `RuleSet.angular()`, `strong()`,
`electromagnetic()`, and `weak()`. Methods return modified copies, so a base
policy can be reused:

```python
strict = rules.with_unknown_policy("isospin", "reject")
diagnostic = strict.disable("isospin_projection")
```

Unknown-input policies are `"allow"`, `"warn"`, and `"reject"`. An allowed
unknown is not the same as a proven pass; inspect `RuleCheck.outcome` and
`missing` when particle metadata are incomplete.

## Generate allowed partial waves

`SelectionRules` couples the daughter spins, scans integral orbital momenta
through `max_l`, checks coupling to the parent spin, and applies its `RuleSet`:

```python
selection = ld.SelectionRules.strong(ld.L(4))
waves = selection.allowed_partial_waves(rho, pi_plus, pi_minus)

for allowed in waves:
    print(
        allowed.wave.label,
        allowed.wave.j,
        allowed.wave.l,
        allowed.wave.s,
        allowed.parity,
        allowed.c_parity,
    )
```

Construct and validate a single wave directly with
`ld.PartialWave(J, L, S)`. The familiar helpers remain available on the
quantum-number objects:

```python
daughter_spins = ld.S(0).coupled_with(ld.S(1))
assert ld.J(1).can_couple_to(ld.L(1), ld.S(0))
```

Rule sets are deliberately explicit about physics assumptions. In particular,
C parity and G parity are not enabled blindly by the strong preset because they
are meaningful only for appropriate eigenstates and multiplets.
