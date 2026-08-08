# Selection rules and partial waves

Selection rules answer two different questions: whether a proposed two-body
decay is allowed under a stated interaction model, and which $(J,L,S)$ waves
should be considered. They do not construct the dynamical amplitude.

## Test a proposed decay

Assume `x_state` and `kaon` are the particles constructed in
{doc}`quantum-numbers`. A rule set makes the physics assumptions explicit:

```python
rules = ld.RuleSet.strong()
report = rules.evaluate(
    x_state,
    daughter_a=kaon,
    daughter_b=kaon,
    l=ld.L(2),
    s=ld.S(0),
)

if not report.is_allowed:
    failed_rules = [
        check.rule
        for check in report.checks
        if check.outcome == "fail"
    ]
```

Standard constructors are `RuleSet.angular()`, `strong()`,
`electromagnetic()`, and `weak()`. Returned rule sets are immutable variants:

```python
strict = rules.with_unknown_policy("isospin", "reject")
without_c = strict.disable("c_parity")
```

Unknown-input policies are `"allow"`, `"warn"`, and `"reject"`. An allowed
unknown is not a demonstrated conservation law; inspect each check when
particle metadata are incomplete.

## Enumerate candidate waves

`SelectionRules` couples daughter spins, scans orbital angular momentum through
`max_l`, checks coupling to the parent spin, and applies the rule set:

```python
selection = ld.SelectionRules.strong(max_l=ld.L(4))
allowed = selection.allowed_partial_waves(
    x_state,
    daughter_a=kaon,
    daughter_b=kaon,
)

waves = [item.wave for item in allowed]
labels = [wave.label for wave in waves]
```

Construct a specific hypothesis with `ld.PartialWave(J, L, S)`. The resulting
wave list supplies discrete model choices; masses, widths, and complex
production couplings remain continuous parameters introduced in
{doc}`expressions`.
