# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.3](https://github.com/denehoffman/laddu/compare/v0.1.2...v0.1.3) - 2024-10-22

### Added

- add options to the minimization callables and add binned `Dataset` loading to Python API
- add filtered and binned loading for `Dataset`s
- export `Status` and `Bound` structs from `ganesh` as PyO3 objects and update `minimize` method accordingly
- add `Debug` derive for `ParameterID`
- add `LadduError` struct and work in proper error forwarding for reading data and registering `Amplitude`s
- use `AsRef` generics to allow more versatile `Variable` construction
- add `ganesh` integration via L-BFGS-B algorithm
- update to latest `PyO3` version

### Fixed

- missed one fully qualified path
- correct some namespace paths
- add `Dataset` and `Event` to `variables`
- add scalar-like `Amplitude`s to python namespace
- reorder expression and parameters
- remove main.rs from tracking

### Other

- update minimization example in README.md
- fix doctest
- update ganesh version
- switch order of expression and parameters in evaluate and project methods

## [0.1.2](https://github.com/denehoffman/laddu/compare/v0.1.1...v0.1.2) - 2024-10-17

### Other

- remove tag check

## [0.1.1](https://github.com/denehoffman/laddu/compare/v0.1.0...v0.1.1) - 2024-10-17

### Other

- remove coverage for f32 feature (for now)
- remove build for 32-bit Windows due to issue with rust-numpy
