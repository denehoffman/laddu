# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0](https://github.com/denehoffman/laddu/compare/laddu-python-v0.2.5...laddu-python-v0.3.0) - 2025-02-21

### Added

- update MPI code to use root-node-agnostic methods
- first pass implementation of MPI interface

### Fixed

- forgot to update the `laddu-python` `use_mpi` function to have a trigger arg
- add feature flag to `laddu-python` and update MSRV for `mpisys` compatibility

### Other

- update all documentation to include MPI modules
- add some clippy lints and clean up some unused imports and redundant code
- *(vectors)* use custom type for 3/4-vectors rather than trait impl for nalgebra Vectors

## [0.2.5](https://github.com/denehoffman/laddu/compare/laddu-python-v0.2.4...laddu-python-v0.2.5) - 2025-01-28

### Other

- update Cargo.toml dependencies

## [0.2.4](https://github.com/denehoffman/laddu/compare/laddu-python-v0.2.3...laddu-python-v0.2.4) - 2025-01-27

### Added

- allow for the use of `sum(list[Dataset])` in Python code

## [0.2.3](https://github.com/denehoffman/laddu/compare/laddu-python-v0.2.2...laddu-python-v0.2.3) - 2025-01-24

### Other

- updated the following local packages: laddu-core
