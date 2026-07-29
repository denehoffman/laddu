from __future__ import annotations

from typing import TYPE_CHECKING

from yamloom import (
    Environment,
    Events,
    Job,
    PullRequestEvent,
    PushEvent,
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

MSRV = '1.94.0'
MINIMUM_PYTHON = '3.11'
PYTHON_MANIFESTS = {
    'laddu': 'python/laddu/Cargo.toml',
    'laddu-local': 'python/laddu-local/Cargo.toml',
}
PYTHON_VERSIONS = ('3.11', '3.12', '3.13', '3.14', '3.14t')
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
            args=('--release', '--out', 'dist', '--generate-stubs'),
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


build_condition = context.github.ref.startswith('refs/tags/')
trusted_context = context.github.event_name != 'pull_request'
release_build_jobs = maturin_jobs(
    needs=['build-check-test', 'free-threaded-test'],
    condition=build_condition,
)

python_release_workflow = Workflow(
    name='Build and Release laddu (Python)',
    on=Events(
        push=PushEvent(branches=['development', 'main'], tags=['v*']),
        pull_request=PullRequestEvent(),
    ),
    jobs={
        'build-check-test': Job(
            name='Build, lint, and test',
            runs_on='ubuntu-latest',
            steps=[
                Checkout(),
                SetupRust(toolchain=MSRV, components=['clippy', 'rustfmt']),
                SetupUV(python_version=MINIMUM_PYTHON),
                SetupMPI(),
                script('cargo fmt --all -- --check'),
                script(
                    'cargo clippy --workspace --all-targets --all-features '
                    '--exclude laddu-python '
                    '--exclude laddu-python-local '
                    '--exclude laddu-python-mpi '
                    '-- -D warnings'
                ),
                script(
                    'cargo test --workspace '
                    '--exclude laddu-python '
                    '--exclude laddu-python-local '
                    '--exclude laddu-python-mpi'
                ),
                script('cargo check -p laddu-python -p laddu-python-local -p laddu-python-mpi'),
                script('uv sync --frozen --inexact --no-install-project --project python/laddu'),
                script(
                    'uv run --no-sync --project python/laddu '
                    'maturin develop --manifest-path python/laddu/Cargo.toml '
                    '--release --generate-stubs'
                ),
                script('uv run --no-sync --project python/laddu ruff check . --exclude=.yamloom.py'),
                script('uv run --no-sync --project python/laddu ty check python docs crates/laddu/examples'),
                script(
                    'uv run --no-sync --project python/laddu python -m unittest discover -s python/tests -p "test_*.py"'
                ),
                script('uv pip install --python python/laddu/.venv/bin/python -r docs/requirements.txt'),
                script(
                    'uv run --no-sync --project python/laddu '
                    'sphinx-build -E -W --keep-going -b html '
                    'docs docs/_build/html'
                ),
            ],
        ),
        'free-threaded-test': Job(
            name='Free-threaded Python',
            runs_on='ubuntu-latest',
            steps=[
                Checkout(),
                SetupRust(toolchain=MSRV),
                SetupUV(python_version='3.14t'),
                script('uv sync --frozen --inexact --no-install-project --project python/laddu'),
                script(
                    'uv run --no-sync --project python/laddu '
                    'maturin develop --manifest-path python/laddu/Cargo.toml '
                    '--release --generate-stubs'
                ),
                script(
                    'uv run --no-sync --project python/laddu python -m unittest discover -s python/tests -p "test_*.py"'
                ),
            ],
        ),
        **release_build_jobs,
        'release': Job(
            name='Publish Python distributions',
            runs_on='ubuntu-22.04',
            needs=list(release_build_jobs),
            environment=Environment('pypi'),
            steps=[
                DownloadArtifact(path='dist', merge_multiple=True),
                PypiPublish(packages_dir='dist'),
            ],
        ),
        'publish-rust': Job(
            name='Publish Rust crates',
            runs_on='ubuntu-latest',
            needs=['release'],
            steps=[
                Checkout(fetch_depth=0),
                SetupMPI(),
                SetupRust(),
                InstallRustTool(tool=['cargo-workspaces']),
                script(f'cargo workspaces publish --from-git --token {context.secrets.CARGO_REGISTRY_TOKEN} --yes'),
            ],
        ),
    },
)

test_build_workflow = Workflow(
    name='Build laddu (Python)',
    on=Events(workflow_dispatch=WorkflowDispatchEvent()),
    jobs=maturin_jobs(upload=False),
)

release_please_workflow = Workflow(
    name='Release Please',
    on=Events(push=PushEvent(branches=['main'])),
    jobs={
        'release-please': Job(
            runs_on='ubuntu-latest',
            steps=[
                ReleasePlease(id='release', token=context.secrets.RELEASE_PLEASE),
            ],
        )
    },
)

benchmark_workflow = Workflow(
    name='CodSpeed Benchmarks',
    on=Events(
        push=PushEvent(branches=['development', 'main']),
        pull_request=PullRequestEvent(),
        workflow_dispatch=WorkflowDispatchEvent(),
    ),
    jobs={
        'benchmarks': Job(
            name='Run benchmarks',
            runs_on='ubuntu-latest',
            steps=[
                Checkout(),
                SetupRust(toolchain=MSRV),
                InstallRustTool(tool=['cargo-codspeed']),
                script(
                    'cargo codspeed build --workspace --benches',
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
                script(
                    'cargo codspeed run',
                    name='Run benchmarks without upload',
                    condition=~trusted_context,
                ),
            ],
        )
    },
)

coverage_workflow = Workflow(
    name='Coverage',
    on=Events(
        push=PushEvent(
            branches=['development', 'main'],
            paths=['**.rs', 'Cargo.toml', 'crates/**/Cargo.toml'],
        ),
        pull_request=PullRequestEvent(paths=['**.rs', 'Cargo.toml', 'crates/**/Cargo.toml']),
        workflow_dispatch=WorkflowDispatchEvent(),
    ),
    jobs={
        'coverage-rust': Job(
            name='Rust coverage',
            runs_on='ubuntu-latest',
            env={'CARGO_TERM_COLOR': 'always'},
            steps=[
                Checkout(),
                SetupRust(toolchain=MSRV),
                SetupMPI(),
                InstallRustTool(tool=['cargo-llvm-cov']),
                script(
                    'cargo llvm-cov --workspace '
                    '--exclude laddu-python '
                    '--exclude laddu-python-local '
                    '--exclude laddu-python-mpi '
                    '--codecov --output-path coverage.json'
                ),
                Codecov(
                    token=context.secrets.CODECOV_TOKEN,
                    files='coverage.json',
                    fail_ci_if_error=True,
                    verbose=True,
                    root_dir=context.github.workspace,
                    condition=trusted_context,
                ),
            ],
        )
    },
)

if __name__ == '__main__':
    sync(
        {
            'benchmark.yml': benchmark_workflow,
            'coverage.yml': coverage_workflow,
            'python-release.yml': python_release_workflow,
            'release-please.yml': release_please_workflow,
            'test-build.yml': test_build_workflow,
        }
    )
