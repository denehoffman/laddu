# Public API compatibility policy

The Rust crates under `crates/` are versioned together. Their default and
all-feature public APIs are recorded in `public-api/*.default.txt` and
`public-api/*.all-features.txt`, respectively. Run
`scripts/public-api.sh --check` before review. When an intentional API change is
approved, regenerate the snapshots with `scripts/public-api.sh --update` and
review the textual diff alongside the implementation.

Snapshots use `cargo-public-api` 0.52.0 and the pinned Rust toolchain declared
in `scripts/public-api.sh` so toolchain output changes do not create incidental
diffs.

Pull requests also run `cargo-semver-checks --all-features` against their base
branch. Snapshot diffs make additive and path-level changes visible; the
semantic check rejects changes that are incompatible with the current workspace
version.

## Public enum changes

Before adding, removing, or changing a public enum variant, verify each relevant
item below. `laddu-expr` has an exhaustive variant inventory test, but downstream
semantic behavior still requires review.

- Update serialization fixtures and document any serialized-form change.
- Update expression facts, costs, structural keys, traversal, and remapping.
- Update scalar and cache lowering, runtime evaluation, and autodiff rules.
- Update CPU, JIT, and WGPU lowering or explicitly reject unsupported variants.
- Regenerate public API snapshots and run `cargo-semver-checks`.
- Treat variant removal or mutation, and additions to exhaustive enums, as a
  compatibility decision that may require a version-policy change.
