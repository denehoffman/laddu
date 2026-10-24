# Repository setup

Complete the rollout below after merging the generated workflows into `main`.

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

- Run the `CI` workflow manually on `main` once and confirm that its full
  correctness route succeeds. Every PR, including metadata-only release PRs,
  runs these checks again against its own source and locked dependencies.
- Protect `main`, require pull requests and up-to-date branches, and require
  the aggregate check named `CI` from GitHub Actions. Apply the requirement
  only after that check exists on `main`; remove obsolete required check names.
- Keep coverage and CodSpeed outside required checks. They run for relevant
  Rust, dependency, benchmark, and CI-tooling changes on PRs and `main`, but
  skip metadata-only releases. Measurement and upload failures remain visible
  without blocking merges or publication.
- Allow **Rebase and merge** and enable **Automatically delete head branches**.
  Rebase merging preserves individual conventional commit messages, including
  `!` and `BREAKING CHANGE` markers used by Release Please. Conflicts still
  require a local update and another successful CI run.
- Confirm that the default branch is `main` and that release pull requests
  target `main`.

## Development and release flow

1. Push a development branch and open a PR. Full correctness checks run on the
   PR, including Rust, standard/free-threaded Python, typing, and documentation.
   Subsequent PR updates cancel superseded CI runs. Pushes alone do not run the
   suite; use manual `CI` dispatch when feedback is needed before opening a PR.
2. Review the changes and choose **Rebase and merge** after `CI` succeeds.
   GitHub deletes eligible remote head branches automatically. Local cleanup
   remains local: switch to `main`, pull with `--ff-only`, and fetch with
   `--prune`. Inspect and delete the old local branch explicitly; after a rebase
   merge Git may not recognize it as merged, so do not automate forced deletion.
3. Release Please opens or updates its release PR. Full correctness checks run
   even when the diff contains only release-managed metadata. Additional release
   checks require the same-repository release branch and the
   `autorelease: pending` label or generated release body (present before GitHub
   attaches labels).
4. Release PRs check lockstep versions, metadata, and semver against the released
   tag recorded in the base branch's manifest. Linux packaging checks build and
   install ordinary/free-threaded wheels and the MPI sdist, exercising the
   installed artifact contracts. Breaking APIs are allowed when the proposed
   version bump permits them; ordinary development PRs do not enforce semver.
   Smoke artifacts use the development profile to avoid full release LTO;
   optimized release artifacts are built on the tag.
5. Review and merge the release PR. Release Please creates the release/tag;
   `python-release.yml` checks tag/version agreement and semver against the
   preceding ancestor release, builds the full platform matrix, and publishes
   to PyPI and crates.io independently. It does not repeat the full test suite.

The publication workflow also supports manual dispatch for building inspectable
artifacts from the selected ref. **Manual dispatch never publishes**, even if
the selected ref is a tag. To recover a failed publication, rerun the failed jobs
on the original tag-push run; check registry state before retrying a partially
completed publisher. Artifact builds must all succeed before either publisher
starts, but success in one registry is not rolled back if the other fails.

Public API snapshots are no longer maintained. Rust API compatibility is
enforced at release time by `cargo-semver-checks`; Python behavior remains
covered by tests and installed artifact checks. This does not detect every
behavioral or serialized-format compatibility change, which still needs review.

The `WGPU hardware validation` workflow is manual and requires a self-hosted
runner labeled `wgpu`. Before an accelerator-affecting release, exercise the
appropriate real-hardware/backend combinations and record the results, as
required by ADR-0008. Workspace CI already covers hardware-independent WGPU
tests.

## Editing workflows

Edit `.yamloom.py`, then run `yamloom sync` to update the owned YAML files and
remove stale workflows. CI uses `yamloom==0.6.1` and `actionlint==1.7.12` to
validate the generated files, plus standard-library tests under
`scripts/tests/test_ci.py`. Run these checks locally with:

```sh
yamloom check
actionlint
python3 -m unittest discover -s scripts/tests -p 'test_ci*.py'
```

The classifier treats unknown files and unsupported metadata changes
conservatively. The aggregate check fails when an applicable check fails, is
cancelled, or unexpectedly skips. Keep its dependency list and the tested gate
policy in sync when adding or renaming CI jobs. Full Rust and standard/free-threaded
Python correctness jobs are mandatory on every route; the gate rejects a
release-only route even if classification accidentally selects one.

## Documentation

- Point Read the Docs at `.readthedocs.yaml`.
- Confirm that Read the Docs builds pull requests and the `main` branch.

## First release

- Keep the `"."` entry in `.release-please-manifest.json` at the currently
  published `laddu` version before enabling the Release Please workflow.
- Keep every workspace package on `version.workspace = true` and declare local
  `laddu` dependencies through the root `[workspace.dependencies]` table. Release
  Please updates that single version source, its dependency requirements, and
  the matching `Cargo.lock` package entries together.
- Confirm that all Rust crate owners and the PyPI trusted publisher are current.
- Merge the generated Release Please pull request only after its version,
  changelog, Cargo manifests, Python metadata, and `CITATION.cff` agree.
