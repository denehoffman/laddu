# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.8.0](https://github.com/denehoffman/laddu/compare/laddu-python-v0.7.1...laddu-python-v0.8.0) - 2025-06-17

### Added

- add `evaluate(Variable)` method to `Dataset` and `Event`

### Other

- [**breaking**] change `Variable` trait methods to operate on `&Dataset` rather than `&Arc<Dataset>`

## [0.7.1](https://github.com/denehoffman/laddu/compare/laddu-python-v0.7.0...laddu-python-v0.7.1) - 2025-05-30

### Other

- updated the following local packages: laddu-core

## [0.7.0](https://github.com/denehoffman/laddu/compare/laddu-python-v0.6.2...laddu-python-v0.7.0) - 2025-05-20

### Added

- improvements to `Dataset` conversions and opening methods

### Fixed

- update Python vector names (closes #57)

## [0.6.2](https://github.com/denehoffman/laddu/compare/laddu-python-v0.6.1...laddu-python-v0.6.2) - 2025-05-16

### Added

- add method for boosting an event or a dataset to a given rest frame

## [0.6.1](https://github.com/denehoffman/laddu/compare/laddu-python-v0.6.0...laddu-python-v0.6.1) - 2025-04-25

### Added

- add available_parallelism function to python API

## [0.6.0](https://github.com/denehoffman/laddu/compare/laddu-python-v0.5.1...laddu-python-v0.6.0) - 2025-04-11

### Other

- complete compatibility with newest version of bincode, remove unused dependencies and features across all crates

## [0.5.1](https://github.com/denehoffman/laddu/compare/laddu-python-v0.5.0...laddu-python-v0.5.1) - 2025-04-04

### Other

- update Cargo.toml dependencies

## [0.4.2](https://github.com/denehoffman/laddu/compare/laddu-python-v0.4.1...laddu-python-v0.4.2) - 2025-03-13

### Added

- add unit-valued `Expression` and define convenience methods for summing and multiplying lists of `Amplitude`s
- add `Debug` and `Display` to every `Variable` (and require them for new ones)

### Fixed

- improve summation and product methods to only return a Zero or One if the list is empty

## [0.4.1](https://github.com/denehoffman/laddu/compare/laddu-python-v0.4.0...laddu-python-v0.4.1) - 2025-03-04

### Fixed

- get rid of unused variable warning

3# [0.4.0](https://github.com/denehoffman/laddu/compare/laddu-python-v0.3.0...laddu-python-v0.3.1) - 2025-02-28

### Added

- redefine eps->aux in `Event` definition

### Other

- move all MPI code to `laddu-python` to make sure MPI docs build properly

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
- _(vectors)_ use custom type for 3/4-vectors rather than trait impl for nalgebra Vectors

## [0.2.5](https://github.com/denehoffman/laddu/compare/laddu-python-v0.2.4...laddu-python-v0.2.5) - 2025-01-28

### Other

- update Cargo.toml dependencies

## [0.2.4](https://github.com/denehoffman/laddu/compare/laddu-python-v0.2.3...laddu-python-v0.2.4) - 2025-01-27

### Added

- allow for the use of `sum(list[Dataset])` in Python code

## [0.2.3](https://github.com/denehoffman/laddu/compare/laddu-python-v0.2.2...laddu-python-v0.2.3) - 2025-01-24

### Other

- updated the following local packages: laddu-core
