# Public API compatibility policy

The Rust crates under `crates/` are versioned together. Review public API changes
alongside their implementation and tests. Breaking changes use conventional
commit `!` markers or `BREAKING CHANGE` footers so Release Please proposes the
appropriate version bump. Checked-in API snapshots are no longer maintained.

Ordinary development PRs may introduce intentional breaking APIs before the
workspace version is bumped. Release PRs run `cargo-semver-checks` 0.50.0 with
`--all-features` against the previous released tag recorded in their base
branch's release manifest. Publication checks the tag against the preceding
ancestor release. Neither check uses `main` as the API baseline, since it already
contains the unreleased changes.

The semantic check uses the actual proposed versions. Before 1.0, an increment
of the minor version permits breaking changes; after 1.0, those changes require
a major bump. Newly introduced crates have no prior API to compare and are
reported separately. Compatibility tooling failures and insufficient version
bumps fail release validation. Python behavior and native distribution contents
are covered by tests and installed artifact checks. Behavioral and serialized
format compatibility still require review beyond Rust API analysis.

## Public enum changes

Before adding, removing, or changing a public enum variant, verify each relevant
item below. `laddu-expr` has an exhaustive variant inventory test, but downstream
semantic behavior still requires review.

- Update serialization fixtures and document any serialized-form change.
- Update expression facts, costs, structural keys, traversal, and remapping.
- Update scalar and cache lowering, runtime evaluation, and autodiff rules.
- Update CPU, JIT, and WGPU lowering or explicitly reject unsupported variants.
- Cover changed contracts with tests and confirm the proposed release version
  permits the API change when release validation runs.
- Treat variant removal or mutation, and additions to exhaustive enums, as a
  compatibility decision that may require a version-policy change.
