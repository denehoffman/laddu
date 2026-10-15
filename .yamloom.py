from __future__ import annotations

from typing import TYPE_CHECKING

from yamloom import (
    Concurrency,
    Environment,
    Events,
    Job,
    Matrix,
    Permissions,
    PullRequestEvent,
    PushEvent,
    Strategy,
    Workflow,
    WorkflowDispatchEvent,
    action,
    script,
    sync,
)
from yamloom.actions.ci.coverage import Codecov
from yamloom.actions.github.artifacts import DownloadArtifact
from yamloom.actions.github.release import ReleasePlease
from yamloom.actions.github.scm import Checkout
from yamloom.actions.packaging.python import PypiPublish
from yamloom.actions.toolchains.python import SetupUV
from yamloom.actions.toolchains.rust import InstallRustTool, SetupRust
from yamloom.actions.toolchains.system import SetupMPI
from yamloom.expressions import BooleanExpression, context
from yamloom.workflows.maturin import (
    MaturinBuildSuite,
    MaturinPlatform,
    MaturinTarget,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

MSRV = '1.96.0'
MINIMUM_PYTHON = '3.11'
PYTHON_MANIFESTS = {
    'laddu': 'python/laddu/Cargo.toml',
    'laddu-local': 'python/laddu-local/Cargo.toml',
}
PYTHON_VERSIONS = ('3.11', '3.14t')
PYTHON_ABI_FEATURES = 'pyo3/abi3-py311,pyo3/abi3t-py315'
WHEEL_PLATFORMS = (
    MaturinPlatform(
        'linux',
        'Build Linux wheels',
        (
            MaturinTarget('ubuntu-22.04', 'x86_64'),
            MaturinTarget('ubuntu-22.04', 'aarch64'),
        ),
        '2014',
    ),
    MaturinPlatform(
        'windows',
        'Build Windows wheels',
        (MaturinTarget('windows-latest', 'x64', 'x64'),),
    ),
    MaturinPlatform(
        'macos',
        'Build macOS wheels',
        (
            MaturinTarget('macos-15-intel', 'x86_64'),
            MaturinTarget('macos-latest', 'aarch64'),
        ),
    ),
)


def maturin_jobs(
    *,
    needs: Sequence[str] | None = None,
    condition: BooleanExpression | None = None,
    upload: bool = True,
) -> dict[str, Job]:
    jobs: dict[str, Job] = {}
    for package, manifest in PYTHON_MANIFESTS.items():
        suite = MaturinBuildSuite(
            package_name=package,
            artifact_prefix=package,
            manifest_path=manifest,
            python_versions=PYTHON_VERSIONS,
            platforms=WHEEL_PLATFORMS,
            args=(
                '--release',
                '--locked',
                '--out',
                'dist',
                '--generate-stubs',
                '--features',
                PYTHON_ABI_FEATURES,
            ),
            needs=needs,
            condition=condition,
            upload=upload,
            sccache=~context.github.ref.startswith('refs/tags/'),
        )
        jobs.update({f'{package}-{name}': job for name, job in suite.jobs().items()})

    mpi_suite = MaturinBuildSuite(
        package_name='laddu-mpi',
        artifact_prefix='laddu-mpi',
        manifest_path='python/laddu-mpi/Cargo.toml',
        python_profile='cpython',
        minimum_python=MINIMUM_PYTHON,
        platforms=(),
        needs=needs,
        condition=condition,
        upload=upload,
    )
    jobs.update({f'laddu-mpi-{name}': job for name, job in mpi_suite.jobs().items()})
    return jobs


# The required workflow always starts; job conditions select a validation route.
READ = Permissions(contents='read')
FULL = "needs.changes.outputs.full == 'true'"
RELEASE = "needs.changes.outputs.release == 'true'"
TAG_PUSH = "github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')"
PR_CONCURRENCY = Concurrency(
    '${{ github.workflow }}-${{ github.event.pull_request.number || github.ref }}',
    cancel_in_progress=context.github.event_name == 'pull_request',
)
RUST_EXCLUDES = '--exclude laddu-python --exclude laddu-python-local --exclude laddu-python-mpi'
UV_SYNC = 'uv sync --frozen --inexact --no-install-project --project python/laddu'
DEVELOP = (
    'uv run --no-sync --directory python/laddu maturin develop '
    '--manifest-path Cargo.toml --locked --release --generate-stubs'
)


def changes_job() -> Job:
    return Job(
        name='Classify changes',
        runs_on='ubuntu-latest',
        outputs={key: f'${{{{ steps.changes.outputs.{key} }}}}' for key in ('full', 'release', 'metrics')},
        steps=[
            Checkout(fetch_depth=0),
            script('python3 scripts/ci.py classify', id='changes'),
        ],
    )


def release_check(*, needs: list[str] | None = None, condition: str | None = None) -> Job:
    return Job(
        name='Release versions and semver',
        runs_on='ubuntu-latest',
        needs=needs,
        condition=condition,
        outputs={'baseline': '${{ steps.release.outputs.baseline }}'},
        steps=[
            Checkout(fetch_depth=0),
            SetupRust(toolchain=MSRV, cache_key='semver'),
            SetupMPI(),
            script('python3 scripts/ci.py release', id='release'),
            InstallRustTool(tool=['cargo-semver-checks@0.50.0']),
            script(
                'python3 scripts/ci.py semver --baseline "$RELEASE_BASELINE"',
                env={'RELEASE_BASELINE': '${{ steps.release.outputs.baseline }}'},
            ),
        ],
    )


ci_workflow = Workflow(
    name='CI',
    on=Events(pull_request=PullRequestEvent(), workflow_dispatch=WorkflowDispatchEvent()),
    permissions=READ,
    concurrency=PR_CONCURRENCY,
    jobs={
        'changes': changes_job(),
        'workflow-check': Job(
            name='Workflow configuration',
            runs_on='ubuntu-latest',
            steps=[
                Checkout(),
                SetupUV(python_version=MINIMUM_PYTHON),
                script('uvx --from yamloom==0.6.1 yamloom check'),
                script(
                    'mkdir -p "$RUNNER_TEMP/actionlint"\n'
                    'curl -fsSL --retry 3 https://github.com/rhysd/actionlint/releases/download/v1.7.12/actionlint_1.7.12_linux_amd64.tar.gz -o "$RUNNER_TEMP/actionlint.tar.gz"\n'
                    "printf '%s  %s\\n' '8aca8db96f1b94770f1b0d72b6dddcb1ebb8123cb3712530b08cc387b349a3d8' \"$RUNNER_TEMP/actionlint.tar.gz\" | sha256sum --check\n"
                    'tar -xzf "$RUNNER_TEMP/actionlint.tar.gz" -C "$RUNNER_TEMP/actionlint" actionlint\n'
                    'echo "$RUNNER_TEMP/actionlint" >> "$GITHUB_PATH"',
                    name='Install actionlint',
                ),
                script('actionlint'),
                script('python3 -m unittest discover -s scripts/tests -p "test_ci*.py"'),
            ],
        ),
        'rust': Job(
            name='Rust correctness',
            runs_on='ubuntu-latest',
            needs=['changes'],
            condition=FULL,
            steps=[
                Checkout(),
                SetupRust(toolchain=MSRV, components=['clippy', 'rustfmt']),
                SetupMPI(),
                script('cargo fmt --all -- --check'),
                script(
                    f'cargo clippy --locked --workspace --all-targets --all-features {RUST_EXCLUDES} -- -D warnings'
                ),
                script(f'cargo test --locked --workspace {RUST_EXCLUDES}'),
                script(
                    'cargo metadata --locked --no-deps --format-version 1 | '
                    "jq -e '([.packages[].version] | unique | length == 1) and "
                    'all(.packages[].dependencies[]; .path == null or .req != "*")\'',
                    name='Validate workspace versions',
                ),
                script('cargo check --locked -p laddu-python -p laddu-python-local -p laddu-python-mpi'),
            ],
        ),
        'python': Job(
            name='Python correctness and documentation',
            runs_on='ubuntu-latest',
            needs=['changes'],
            condition=FULL,
            steps=[
                Checkout(),
                SetupRust(toolchain=MSRV, cache_key='python-3.11'),
                SetupUV(python_version=MINIMUM_PYTHON),
                SetupMPI(),
                script(UV_SYNC),
                script(DEVELOP),
                script('uv run --no-sync --project python/laddu ruff check . --exclude=.yamloom.py'),
                script('uv run --no-sync --project python/laddu ty check python docs crates/laddu/examples'),
                script('uv run --no-sync --project python/laddu python -m pytest python/tests'),
                script('uv pip install --python python/laddu/.venv/bin/python -r docs/requirements.txt'),
                script(
                    'uv run --no-sync --project python/laddu sphinx-build -E -W --keep-going -b html docs docs/_build/html'
                ),
            ],
        ),
        'free-threaded': Job(
            name='Free-threaded Python',
            runs_on='ubuntu-latest',
            needs=['changes'],
            condition=FULL,
            steps=[
                Checkout(),
                SetupRust(toolchain=MSRV, cache_key='python-3.14t'),
                SetupUV(python_version='3.14t'),
                script(UV_SYNC),
                script(DEVELOP),
                script('uv run --no-sync --project python/laddu python -m pytest python/tests'),
            ],
        ),
        'release-check': release_check(needs=['changes'], condition=RELEASE),
        'release-smoke': Job(
            name='Release Linux packaging (${{ matrix.python }})',
            runs_on='ubuntu-latest',
            needs=['changes', 'release-check'],
            condition=RELEASE,
            strategy=Strategy(matrix=Matrix(python=['3.11', '3.14t']), fast_fail=False),
            steps=[
                Checkout(),
                SetupRust(toolchain=MSRV, cache_key='smoke-${{ matrix.python }}'),
                SetupUV(python_version='${{ matrix.python }}'),
                SetupMPI(),
                script('python3 scripts/ci_smoke.py', env={'SMOKE_PYTHON': '${{ matrix.python }}'}),
            ],
        ),
        'ci': Job(
            name='CI',
            runs_on='ubuntu-latest',
            needs=['changes', 'workflow-check', 'rust', 'python', 'free-threaded', 'release-check', 'release-smoke'],
            condition='always()',
            steps=[
                Checkout(),
                script('python3 scripts/ci.py gate', env={'CI_NEEDS': '${{ toJSON(needs) }}'}),
            ],
        ),
    },
)

# Dispatch builds artifacts for inspection; only an actual tag push can publish.
release_build_jobs = maturin_jobs(needs=['preflight'])
python_release_workflow = Workflow(
    name='Build and publish laddu',
    on=Events(push=PushEvent(tags=['v*']), workflow_dispatch=WorkflowDispatchEvent()),
    permissions=READ,
    concurrency=Concurrency('${{ github.workflow }}-${{ github.ref }}', cancel_in_progress=False),
    jobs={
        'preflight': Job(
            name='Release preflight',
            runs_on='ubuntu-latest',
            steps=[
                Checkout(fetch_depth=0),
                SetupRust(toolchain=MSRV, cache_key='semver'),
                SetupMPI(condition=TAG_PUSH),
                script('python3 scripts/ci.py release', id='release', condition=TAG_PUSH),
                InstallRustTool(tool=['cargo-semver-checks@0.50.0'], condition=TAG_PUSH),
                script(
                    'python3 scripts/ci.py semver --baseline "$RELEASE_BASELINE"',
                    env={'RELEASE_BASELINE': '${{ steps.release.outputs.baseline }}'},
                    condition=TAG_PUSH,
                ),
            ],
        ),
        **release_build_jobs,
        'release': Job(
            name='Publish Python distributions',
            runs_on='ubuntu-22.04',
            needs=['preflight', *release_build_jobs],
            condition=TAG_PUSH,
            environment=Environment('pypi'),
            steps=[
                DownloadArtifact(path='dist', merge_multiple=True),
                PypiPublish(packages_dir='dist'),
            ],
        ),
        'publish-rust': Job(
            name='Publish Rust crates',
            runs_on='ubuntu-latest',
            needs=['preflight', *release_build_jobs],
            condition=TAG_PUSH,
            steps=[
                Checkout(fetch_depth=0),
                SetupMPI(),
                SetupRust(toolchain=MSRV),
                InstallRustTool(tool=['cargo-workspaces']),
                script(
                    'cargo workspaces publish --from-git --token "$CARGO_REGISTRY_TOKEN" --yes',
                    env={'CARGO_REGISTRY_TOKEN': context.secrets.CARGO_REGISTRY_TOKEN},
                ),
            ],
        ),
    },
)

release_please_workflow = Workflow(
    name='Release Please',
    on=Events(push=PushEvent(branches=['main'])),
    permissions=READ,
    # Never interrupt release creation. Pending pushes may coalesce; the next run
    # still discovers all commits and merged release PRs on main.
    concurrency=Concurrency('release-please-main', cancel_in_progress=False),
    jobs={
        'release-please': Job(
            runs_on='ubuntu-latest',
            steps=[ReleasePlease(id='release', token=context.secrets.RELEASE_PLEASE)],
        ),
    },
)

trusted_context = context.github.event_name != 'pull_request'
metrics_workflow = Workflow(
    name='Coverage and benchmarks',
    on=Events(
        pull_request=PullRequestEvent(),
        push=PushEvent(branches=['main']),
        workflow_dispatch=WorkflowDispatchEvent(),
    ),
    permissions=READ,
    concurrency=PR_CONCURRENCY,
    jobs={
        'changes': changes_job(),
        'coverage': Job(
            name='Rust coverage',
            runs_on='ubuntu-latest',
            needs=['changes'],
            condition="needs.changes.outputs.metrics == 'true'",
            steps=[
                Checkout(),
                SetupRust(toolchain=MSRV, cache_key='coverage'),
                SetupMPI(),
                InstallRustTool(tool=['cargo-llvm-cov']),
                script(f'cargo llvm-cov --locked --workspace {RUST_EXCLUDES} --codecov --output-path coverage.json'),
                Codecov(
                    token=context.secrets.CODECOV_TOKEN,
                    files='coverage.json',
                    fail_ci_if_error=True,
                    root_dir=context.github.workspace,
                    condition=trusted_context,
                ),
            ],
        ),
        'benchmarks': Job(
            name='CodSpeed benchmarks',
            runs_on='ubuntu-latest',
            needs=['changes'],
            condition="needs.changes.outputs.metrics == 'true'",
            steps=[
                Checkout(),
                SetupRust(toolchain=MSRV, cache_key='codspeed'),
                InstallRustTool(tool=['cargo-codspeed']),
                script(
                    'cargo codspeed build --bench generation --bench jit_benchmark '
                    '--bench kmatrix_benchmark --bench projection_benchmark --features laddu/jit',
                    env={'CARGO_BUILD_JOBS': '1'},
                ),
                action(
                    'CodSpeed Action',
                    'CodSpeedHQ/action',
                    ref='v4',
                    with_opts={
                        'mode': 'simulation',
                        'run': 'cargo codspeed run',
                        'token': context.secrets.CODSPEED_TOKEN,
                    },
                    condition=trusted_context,
                ),
                script('cargo codspeed run', name='Run benchmarks without upload', condition=~trusted_context),
            ],
        ),
    },
)

wgpu_workflow = Workflow(
    name='WGPU hardware validation',
    on=Events(workflow_dispatch=WorkflowDispatchEvent()),
    permissions=READ,
    jobs={
        'hardware': Job(
            name='WGPU hardware integration tests',
            runs_on=['self-hosted', 'wgpu'],
            steps=[
                Checkout(),
                SetupRust(toolchain=MSRV),
                script('cargo test --locked -p laddu-wgpu --test hardware -- --ignored'),
            ],
        ),
    },
)

if __name__ == '__main__':
    sync(
        {
            'ci.yml': ci_workflow,
            'metrics.yml': metrics_workflow,
            'python-release.yml': python_release_workflow,
            'release-please.yml': release_please_workflow,
            'wgpu.yml': wgpu_workflow,
        }
    )
