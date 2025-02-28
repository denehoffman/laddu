# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0](https://github.com/denehoffman/laddu/compare/py-laddu-v0.3.0...py-laddu-v0.3.1) - 2025-02-28

### Added

- split `laddu` python package into two, with and without MPI support
- redefine eps->aux in `Event` definition

### Fixed

- reorganize package structure

### Other

- move all MPI code to `laddu-python` to make sure MPI docs build properly
- finalize conversion of eps->aux in data formatting
- fix citation formatting

## [0.3.0](https://github.com/denehoffman/laddu/compare/py-laddu-v0.2.6...py-laddu-v0.3.0) - 2025-02-21

### Added

- make `mpi` a feature in `py-laddu` to allow people to build the python package without it
- update MPI code to use root-node-agnostic methods
- first pass implementation of MPI interface
- switch the MPI implementation to use safe Rust via a RwLock

### Fixed

- add non-MPI failing functions for MPI calls on non-MPI python builds
- add mpi feature for laddu-python to py-laddu
- calling get_world before use_mpi causes errors
- correct the open method and counts/displs methods

### Other

- update all documentation to include MPI modules
- add mpich to builds
- _(vectors)_ complete tests for vectors module
- _(vectors)_ add more vector test coverage
- _(vectors)_ use custom type for 3/4-vectors rather than trait impl for nalgebra Vectors
- add some clippy lints and clean up some unused imports and redundant code
- _(ganesh_ext)_ documenting a few missed functions
- use elided lifetimes

## [0.2.6](https://github.com/denehoffman/laddu/compare/py-laddu-v0.2.5...py-laddu-v0.2.6) - 2025-01-28

### Added

- bump `ganesh` to add "skip_hessian" minimization option to skip calculation of Hessian matrix

### Fixed

- use proper ownership in setting algorithm error mode
- use correct enum in L-BFGS-B error method

### Other

- update Cargo.toml dependencies

## [0.2.5](https://github.com/denehoffman/laddu/compare/py-laddu-v0.2.4...py-laddu-v0.2.5) - 2025-01-27

### Fixed

- move `rayon` feature bounds inside methods to clean up the code and avoid duplication

### Other

- _(data)_ fix bootstrap tests by changing seed
- update dependencies and remove `rand` and `rand_chacha`

## [0.2.4](https://github.com/denehoffman/laddu/releases/tag/py-laddu-v0.2.4) - 2025-01-26

### Added

- add `BinnedGuideTerm` under new `experimental` module
- allow users to add `Dataset`s together to form a new `Dataset`

### Fixed

- fixed python examples and readme paths
- modify tests and workflows to new structure

### Other

- bump py-laddu version
- _(py-laddu)_ release v0.2.3
- manually update py-laddu version
- omit tests and docs in python coverage
- correct path of extensions module
- _(py-laddu)_ release v0.2.0
- release all crates manually
- release-plz does not like the way I've set up the workspace. I've looked at conda/rattler for some inspiration, but I might need to manually publish each crate once to get the ball rolling
- add rust version to py-laddu
- complete python integration to new py-laddu crate
- major rewrite

## [0.2.3](https://github.com/denehoffman/laddu/releases/tag/py-laddu-v0.2.3) - 2025-01-24

### Added

- add `BinnedGuideTerm` under new `experimental` module
- allow users to add `Dataset`s together to form a new `Dataset`

### Fixed

- fixed python examples and readme paths
- modify tests and workflows to new structure

### Other

- manually update py-laddu version
- omit tests and docs in python coverage
- correct path of extensions module
- _(py-laddu)_ release v0.2.0
- release all crates manually
- release-plz does not like the way I've set up the workspace. I've looked at conda/rattler for some inspiration, but I might need to manually publish each crate once to get the ball rolling
- add rust version to py-laddu
- complete python integration to new py-laddu crate
- major rewrite

## [0.2.2](https://github.com/denehoffman/laddu/releases/tag/py-laddu-v0.2.2) - 2025-01-24

### Fixed

- corrected signature in methods that read from AmpTools trees
- fixed python examples and readme paths
- modify tests and workflows to new structure

### Other

- bump version
- _(py-laddu)_ release v0.2.1
- force version bump
- fix python docs to use "extensions" rather than "likelihoods"
- _(py-laddu)_ release v0.2.0
- release all crates manually
- release-plz does not like the way I've set up the workspace. I've looked at conda/rattler for some inspiration, but I might need to manually publish each crate once to get the ball rolling
- add rust version to py-laddu
- complete python integration to new py-laddu crate
- major rewrite

## [0.2.1](https://github.com/denehoffman/laddu/releases/tag/py-laddu-v0.2.1) - 2025-01-24

### Fixed

- corrected signature in methods that read from AmpTools trees
- fixed python examples and readme paths
- modify tests and workflows to new structure

### Other

- force version bump
- fix python docs to use "extensions" rather than "likelihoods"
- _(py-laddu)_ release v0.2.0
- release all crates manually
- release-plz does not like the way I've set up the workspace. I've looked at conda/rattler for some inspiration, but I might need to manually publish each crate once to get the ball rolling
- add rust version to py-laddu
- complete python integration to new py-laddu crate
- major rewrite

## [0.2.0](https://github.com/denehoffman/laddu/releases/tag/py-laddu-v0.2.0) - 2025-01-21

### Fixed

- modify tests and workflows to new structure

### Other

- release all crates manually
- release-plz does not like the way I've set up the workspace. I've looked at conda/rattler for some inspiration, but I might need to manually publish each crate once to get the ball rolling
- add rust version to py-laddu
- complete python integration to new py-laddu crate
- major rewrite
