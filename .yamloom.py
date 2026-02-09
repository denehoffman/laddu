from __future__ import annotations

from dataclasses import dataclass

from yamloom import (
    Environment,
    Events,
    Job,
    Matrix,
    PullRequestEvent,
    PushEvent,
    Strategy,
    Workflow,
    WorkflowCallEvent,
    WorkflowDispatchEvent,
    WorkflowSecret,
    action,
    script,
)
from yamloom.actions.ci.coverage import Codecov
from yamloom.actions.github.artifacts import DownloadArtifact, UploadArtifact
from yamloom.actions.github.release import ReleasePlease
from yamloom.actions.github.scm import Checkout
from yamloom.actions.packaging.python import Maturin
from yamloom.actions.toolchains.python import SetupPython, SetupUV
from yamloom.actions.toolchains.rust import InstallRustTool, SetupRust
from yamloom.actions.toolchains.system import SetupMPI
from yamloom.expressions import context


@dataclass
class Target:
    runner: str
    target: str
    skip_python_versions: list[str] | None = None


DEFAULT_PYTHON_VERSIONS = [
    '3.7',
    '3.8',
    '3.9',
    '3.10',
    '3.11',
    '3.12',
    '3.13',
    '3.13t',
    '3.14',
    '3.14t',
    'pypy3.11',
]


@dataclass
class TargetJob:
    job_name: str
    short_name: str
    targets: list[Target]


TARGET_JOBS_CPU = [
    TargetJob(
        'Build Linux Wheels',
        'linux',
        [
            Target(
                'ubuntu-22.04',
                target,
            )
            for target in [
                'x86_64',
                'x86',
                'aarch64',
                'armv7',
                's390x',
                'ppc64le',
            ]
        ],
    ),
    TargetJob(
        'Build (musl) Linux Wheels',
        'musllinux',
        [
            Target(
                'ubuntu-22.04',
                target,
            )
            for target in [
                'x86_64',
                'x86',
                'aarch64',
                'armv7',
            ]
        ],
    ),
    TargetJob(
        'Build Windows Wheels',
        'windows',
        [
            Target('windows-latest', 'x64'),
            Target('windows-latest', 'x86', ['pypy3.11']),
            Target(
                'windows-11-arm',
                'aarch64',
                ['3.7', '3.8', '3.9', '3.10', '3.11', '3.13t', '3.14t', 'pypy3.11'],
            ),
        ],
    ),
    TargetJob(
        'Build macOS Wheels',
        'macos',
        [
            Target(
                'macos-15-intel',
                'x86_64',
            ),
            Target(
                'macos-latest',
                'aarch64',
            ),
        ],
    ),
]

TARGET_JOBS_MPI = [
    TargetJob(
        'Build Linux Wheels',
        'linux',
        [
            Target(
                'ubuntu-22.04',
                target,
            )
            for target in [
                'x86_64',
                'x86',
                'aarch64',
                'armv7',
                's390x',
                'ppc64le',
            ]
        ],
    ),
    TargetJob(
        'Build (musl) Linux Wheels',
        'musllinux',
        [
            Target(
                'ubuntu-22.04',
                target,
            )
            for target in [
                'x86_64',
                'x86',
                'aarch64',
                'armv7',
            ]
        ],
    ),
    TargetJob(
        'Build Windows Wheels',
        'windows',
        [
            Target('windows-latest', 'x64'),
            Target('windows-latest', 'x86', ['pypy3.11']),
            Target(
                'windows-11-arm',
                'aarch64',
                ['3.7', '3.8', '3.9', '3.10', '3.11', '3.13t', '3.14t', 'pypy3.11'],
            ),
        ],
    ),
    TargetJob(
        'Build macOS Wheels',
        'macos',
        [
            Target(
                'macos-15-intel',
                'x86_64',
            ),
            Target(
                'macos-latest',
                'aarch64',
            ),
        ],
    ),
]


def resolve_python_versions(skip: list[str] | None) -> list[str]:
    if not skip:
        return DEFAULT_PYTHON_VERSIONS
    skipped = set(skip)
    return [version for version in DEFAULT_PYTHON_VERSIONS if version not in skipped]


def create_build_job(
    job_name: str,
    name: str,
    targets: list[Target],
    *,
    mpi: bool,
    needs: list[str] | None = None,
    upload: bool = True,
) -> Job:
    def platform_entry(target: Target) -> dict[str, object]:
        entry = {
            'runner': target.runner,
            'target': target.target,
            'python_versions': resolve_python_versions(target.skip_python_versions),
        }
        python_arch = (
            ('arm64' if target.target == 'aarch64' else target.target)
            if name == 'windows'
            else None
        )
        if python_arch is not None:
            entry['python_arch'] = python_arch
        return entry

    return Job(
        steps=[
            Checkout(),
            script(
                f'printf "%s\n" {context.matrix.platform.python_versions.as_array().join(" ")} >> version.txt',
            ),
            SetupPython(
                python_version_file='version.txt',
                architecture=context.matrix.platform.python_arch.as_str()
                if name == 'windows'
                else None,
            ),
        ]
        + ([SetupMPI()] if mpi else [])
        + [
            Maturin(
                name='Build wheels',
                target=context.matrix.platform.target.as_str(),
                args=f'--release --out dist --manifest-path crates/py-laddu-{"mpi" if mpi else "cpu"}/Cargo.toml --interpreter {context.matrix.platform.python_versions.as_array().join(" ")}',
                sccache=~context.github.ref.startswith('refs/tags/'),
                manylinux='musllinux_1_2'
                if name == 'musllinux'
                else ('auto' if name == 'linux' else None),
            ),
        ]
        + (
            [
                UploadArtifact(
                    path='dist',
                    artifact_name=f'{"mpi" if mpi else "cpu"}-{name}-{context.matrix.platform.target}',
                )
            ]
            if upload
            else []
        ),
        name=f'{job_name} ({"mpi" if mpi else "cpu"})',
        runs_on=context.matrix.platform.runner.as_str(),
        strategy=Strategy(
            fast_fail=False,
            matrix=Matrix(
                platform=[platform_entry(target) for target in targets],
            ),
        ),
        needs=needs,
        condition=context.github.ref.startswith('refs/tags/py-laddu')
        | (context.github.event_name == 'workflow_dispatch'),
    )


test_build_workflow = Workflow(
    name='Build laddu (Python)',
    on=Events(
        workflow_dispatch=WorkflowDispatchEvent(),
    ),
    jobs={
        **{
            f'{tj.short_name}-cpu': create_build_job(
                tj.job_name, tj.short_name, tj.targets, mpi=False, upload=False
            )
            for tj in TARGET_JOBS_CPU
        },
        **{
            f'{tj.short_name}-mpi': create_build_job(
                tj.job_name, tj.short_name, tj.targets, mpi=True, upload=False
            )
            for tj in TARGET_JOBS_CPU
        },
    },
)


python_release_workflow = Workflow(
    name='Build and Release laddu (Python)',
    on=Events(
        push=PushEvent(
            branches=['main'], tags=['py-laddu*', '!py-laddu-cpu*', '!py-laddu-mpi*']
        ),
        pull_request=PullRequestEvent(opened=True, synchronize=True, reopened=True),
        workflow_dispatch=WorkflowDispatchEvent(),
    ),
    jobs={
        'build-check-test': Job(
            steps=[
                Checkout(),
                SetupRust(components=['clippy']),
                SetupUV(python_version='3.10'),
                SetupMPI(),
                script('cargo clippy'),
                InstallRustTool(tool=['cargo-hack']),
                script(
                    'cargo hack check --rust-version --feature-powerset --no-dev-deps'
                ),
                script('cargo hack test --feature-powerset'),
                script(
                    'uv venv',
                    '. .venv/bin/activate',
                    'echo PATH=$PATH >> $GITHUB_ENV',
                    'uvx --with "maturin[patchelf]>=1.7,<2" maturin build --manifest-path py-laddu-cpu/Cargo.toml --release -o py-laddu-cpu/dist',
                    'uv pip install --no-cache-dir --find-links py-laddu-cpu/dist laddu-cpu',
                    'uv pip install --no-cache-dir -e "py-laddu[tests]"',
                ),
                script('uvx ruff check . --extend-exclude=.yamloom.py'),
                script('uvx ty check . --exclude=.yamloom.py'),
                script('uv run pytest'),
            ],
            runs_on='ubuntu-latest',
        ),
        **{
            f'{tj.short_name}-cpu': create_build_job(
                tj.job_name,
                tj.short_name,
                tj.targets,
                needs=['build-check-test'],
                mpi=False,
            )
            for tj in TARGET_JOBS_CPU
        },
        **{
            f'{tj.short_name}-mpi': create_build_job(
                tj.job_name,
                tj.short_name,
                tj.targets,
                needs=['build-check-test'],
                mpi=True,
            )
            for tj in TARGET_JOBS_CPU
        },
        'sdist-cpu': Job(
            steps=[
                Checkout(),
                Maturin(
                    name='Build sdist',
                    command='sdist',
                    args='--out dist --manifest-path py-laddu-cpu/Cargo.toml',
                ),
                UploadArtifact(path='dist', artifact_name='cpu-sdist'),
            ],
            name='Build Source Distribution',
            runs_on='ubuntu-22.04',
            needs=['build-check-test'],
            condition=context.github.ref.startswith('refs/tags/py-laddu')
            | (context.github.event_name == 'workflow_dispatch'),
        ),
        'sdist-mpi': Job(
            steps=[
                Checkout(),
                Maturin(
                    name='Build sdist',
                    command='sdist',
                    args='--out dist --manifest-path py-laddu-mpi/Cargo.toml',
                ),
                UploadArtifact(path='dist', artifact_name='mpi-sdist'),
            ],
            name='Build Source Distribution',
            runs_on='ubuntu-22.04',
            needs=['build-check-test'],
            condition=context.github.ref.startswith('refs/tags/py-laddu')
            | (context.github.event_name == 'workflow_dispatch'),
        ),
        'release': Job(
            steps=[
                DownloadArtifact(),
                SetupUV(),
                script(
                    'uv publish --trusted-publishing always cpu-*/*',
                ),
                script(
                    'uv publish --trusted-publishing always mpi-*/*',
                ),
                script(
                    'uv build py-laddu --out-dir dist',
                    'uv publish --trusted-publishing always dist/*',
                ),
            ],
            name='Release',
            runs_on='ubuntu-22.04',
            condition=context.github.ref.startswith('refs/tags/py-laddu')
            | (context.github.event_name == 'workflow_dispatch'),
            needs=[
                *[f'{tj.short_name}-cpu' for tj in TARGET_JOBS_CPU],
                *[f'{tj.short_name}-mpi' for tj in TARGET_JOBS_MPI],
                'sdist-cpu',
                'sdist-mpi',
            ],
            environment=Environment('pypi'),
        ),
    },
)

release_please_workflow = Workflow(
    name='Release Please',
    on=Events(
        push=PushEvent(
            branches=['main'],
        ),
    ),
    jobs={
        'release-please': Job(
            steps=[
                ReleasePlease(
                    id='release',
                    token=context.secrets.RELEASE_PLEASE,
                ),
                Checkout(
                    condition=ReleasePlease.releases_created(
                        'release'
                    ).from_json_to_bool()
                ),
                SetupRust(
                    condition=ReleasePlease.releases_created(
                        'release'
                    ).from_json_to_bool()
                ),
                InstallRustTool(
                    tool=['cargo-workspaces'],
                    condition=ReleasePlease.releases_created(
                        'release'
                    ).from_json_to_bool(),
                ),
                script(
                    f'cargo workspaces publish --from-git --token {context.secrets.CARGO_REGISTRY_TOKEN} --yes',
                    condition=ReleasePlease.releases_created(
                        'release'
                    ).from_json_to_bool(),
                ),
            ],
            runs_on='ubuntu-latest',
        )
    },
)

benchmark_workflow = Workflow(
    name='CodSpeed Benchmarks',
    on=Events(
        push=PushEvent(branches=['main']),
        pull_request=PullRequestEvent(opened=True, synchronize=True, reopened=True),
        workflow_dispatch=WorkflowDispatchEvent(),
    ),
    jobs={
        'benchmarks': Job(
            steps=[
                Checkout(),
                SetupRust(),
                InstallRustTool(tool=['cargo-codspeed']),
                script('cargo codspeed build'),
                action(
                    'CodSpeed Action',
                    'CodSpeedHQ/action',
                    ref='v4',
                    with_opts={
                        'mode': 'simulation',
                        'run': 'cargo codspeed run',
                        'token': context.secrets.CODSPEED_TOKEN,
                    },
                ),
            ],
            name='Run Benchmarks',
            runs_on='ubuntu-latest',
        )
    },
)

coverage_workflow = Workflow(
    name='Coverage',
    on=Events(
        push=PushEvent(
            branches=['main'], paths=['**.rs', '**.py', '.github/workflows/coverage.yml']
        ),
        pull_request=PullRequestEvent(
            opened=True,
            synchronize=True,
            reopened=True,
            paths=['**.rs', '**.py', '.github/workflows/coverage.yml'],
        ),
        workflow_call=WorkflowCallEvent(
            secrets={'codecov_token': WorkflowSecret(required=True)}
        ),
        workflow_dispatch=WorkflowDispatchEvent(),
    ),
    jobs={
        'coverage-rust': Job(
            steps=[
                Checkout(),
                SetupRust(toolchain='nightly'),
                SetupMPI(),
                InstallRustTool(tool=['cargo-llvm-cov']),
                script(
                    'cargo llvm-cov --workspace --lcov --output-path coverage-rust.lcov --summary-only --exclude-from-report py-laddu'
                ),
                UploadArtifact(path='coverage-rust.lcov', artifact_name='coverage-rust'),
            ],
            runs_on='ubuntu-latest',
            env={'CARGO_TERM_COLOR': 'always'},
        ),
        'coverage-python': Job(
            steps=[
                Checkout(),
                SetupRust(),
                SetupUV(),
                script(
                    'uv venv',
                    '. .venv/bin/activate',
                    'echo PATH=$PATH >> $GITHUB_ENV',
                    'uvx --with "maturin[patchelf]>=1.7,<2" maturin build --manifest-path py-laddu-cpu/Cargo.toml --release -o py-laddu-cpu/dist',
                    'uv pip install --no-cache-dir --find-links py-laddu-cpu/dist laddu-cpu',
                    'uv pip install --no-cache-dir -e "py-laddu[tests]"',
                    'pytest --cov --cov-report xml:coverage-python.xml',
                ),
                UploadArtifact(
                    path='coverage-python.xml', artifact_name='coverage-python'
                ),
            ],
            runs_on='ubuntu-latest',
            env={'CARGO_TERM_COLOR': 'always'},
        ),
        'upload-coverage': Job(
            steps=[
                Checkout(),
                DownloadArtifact(merge_multiple=True),
                Codecov(
                    token=context.secrets.CODECOV_TOKEN,
                    files='coverage-rust.lcov,coverage-python.xml',
                    fail_ci_if_error=True,
                    verbose=True,
                    root_dir=context.github.workspace,
                ),
            ],
            runs_on='ubuntu-latest',
            needs=['coverage-rust', 'coverage-python'],
        ),
    },
)

if __name__ == '__main__':
    test_build_workflow.dump('.github/workflows/test-build.yml')
    python_release_workflow.dump('.github/workflows/python-release.yml')
    release_please_workflow.dump('.github/workflows/release-please.yml')
    benchmark_workflow.dump('.github/workflows/benchmark.yml')
    coverage_workflow.dump('.github/workflows/coverage.yml')
