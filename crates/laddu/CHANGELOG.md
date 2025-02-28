# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.1](https://github.com/denehoffman/laddu/compare/laddu-v0.3.0...laddu-v0.3.1) - 2025-02-28

### Added

- redefine eps->aux in `Event` definition

### Other

- finalize conversion of eps->aux in data formatting
- fix citation formatting

## [0.3.0](https://github.com/denehoffman/laddu/compare/laddu-v0.2.6...laddu-v0.3.0) - 2025-02-21

### Added

- update MPI code to use root-node-agnostic methods
- first pass implementation of MPI interface
- switch the MPI implementation to use safe Rust via a RwLock

### Fixed

- add sentry dependency for which to force the version, not sure what the best fix really is, but this should work for now
- calling get_world before use_mpi causes errors
- correct the open method and counts/displs methods

### Other

- update all documentation to include MPI modules
- add some clippy lints and clean up some unused imports and redundant code
- *(vectors)* use custom type for 3/4-vectors rather than trait impl for nalgebra Vectors
- update benchmark to only run on powers of 2 threads
- *(vectors)* complete tests for vectors module
- *(vectors)* add more vector test coverage
- *(ganesh_ext)* documenting a few missed functions
- use elided lifetimes

## [0.2.6](https://github.com/denehoffman/laddu/compare/laddu-v0.2.5...laddu-v0.2.6) - 2025-01-28

### Added

- bump `ganesh`  to add "skip_hessian" minimization option to skip calculation of Hessian matrix

### Fixed

- use proper ownership in setting algorithm error mode
- use correct enum in L-BFGS-B error method

### Other

- update Cargo.toml dependencies

## [0.2.5](https://github.com/denehoffman/laddu/compare/laddu-v0.2.4...laddu-v0.2.5) - 2025-01-27

### Fixed

- move `rayon` feature bounds inside methods to clean up the code and avoid duplication

### Other

- update dependencies and remove `rand` and `rand_chacha`
- *(data)* fix bootstrap tests by changing seed

## [0.2.4](https://github.com/denehoffman/laddu/compare/laddu-v0.2.3...laddu-v0.2.4) - 2025-01-26

### Added

- implement custom gradient for `BinnedGuideTerm`
- add `project_gradient` and `project_gradient_with` methods to `NLL`

### Other

- fix some docstring links in `laddu-extensions`

## [0.2.3](https://github.com/denehoffman/laddu/compare/laddu-v0.2.2...laddu-v0.2.3) - 2025-01-24

### Added

- add `BinnedGuideTerm` under new `experimental` module
- allow users to add `Dataset`s together to form a new `Dataset`
