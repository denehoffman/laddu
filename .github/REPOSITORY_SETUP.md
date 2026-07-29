# Repository setup

Complete these settings after rebasing this work onto the upstream repository.

## Actions and releases

- Allow GitHub Actions to create and approve pull requests.
- Add a `RELEASE_PLEASE` secret containing a fine-grained personal access token
  or GitHub App token that can create release pull requests, tags, and releases.
  A separate token is required because releases created by `GITHUB_TOKEN` do not
  trigger the tag-driven publication workflow.
- Add the `CARGO_REGISTRY_TOKEN` repository secret and confirm that its owner can
  publish every crate in this workspace.
- Configure a PyPI trusted publisher for each published Python project
  (`laddu`, `laddu-local`, and `laddu-mpi`), using the `pypi` environment,
  repository, and `.github/workflows/python-release.yml` workflow.
- Add `CODECOV_TOKEN` and `CODSPEED_TOKEN` secrets. Pull requests from forks run
  without either upload.

## Branches and checks

- Protect `main` and `development`.
- Require the `Build, lint, and test` and `Free-threaded Python` checks before
  merging.
- Confirm that the default branch is `main` and that release pull requests
  target `main`.

## Documentation

- Point Read the Docs at `.readthedocs.yaml`.
- Confirm that Read the Docs builds pull requests and the `main` branch.

## First release

- Keep `.release-please-manifest.json` at the currently published Laddu version
  before enabling the Release Please workflow.
- Confirm that all Rust crate owners and the PyPI trusted publisher are current.
- Merge the generated Release Please pull request only after its version,
  changelog, Cargo manifests, Python metadata, and `CITATION.cff` agree.
