# Release checks and publishing

The `python-release.yml` workflow checks Rust formatting, linting, tests, Python
types and tests, and the Sphinx documentation on pushes and pull requests. Python
tests run with pytest, which also collects the unittest-style test classes.
Release artifacts are built only for `v*` tags after both the standard and
free-threaded Python checks pass.

Before tagging a release, run the local equivalents:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features \
  --exclude laddu-python --exclude laddu-python-local --exclude laddu-python-mpi -- -D warnings
cargo test --workspace \
  --exclude laddu-python --exclude laddu-python-local --exclude laddu-python-mpi
just check-python-types
just test-python
just docs-build
```

The tag workflow publishes the Python distributions to PyPI first. Once that
job succeeds, `cargo workspaces publish --publish-as-is` publishes the Rust
workspace crates in dependency order. The `laddu-cas-macros` proc macro crate
is a workspace member and a dependency of `laddu-compile`, so it is included
in that publish plan. The Rust publish step requires `CARGO_REGISTRY_TOKEN`.
The workflow does not create another version commit, tag, or push.
