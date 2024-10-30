# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.4](https://github.com/denehoffman/laddu/compare/v0.1.3...v0.1.4) - 2024-10-30

### Added

- add `gen_amp` config file for Python `example_1`
- add python example
- add `Debug` derive for `Parameters`
- add method to input beam polarization info and assume unity weights if none are provided
- adds a `LikelihoodScalar` term that can be used to scale `LikelihoodTerm`s by a scalar-valued parameter
- expose the underlying dataset and Monte-Carlo dataset in the Python API for `NLL` and add method to turn an `NLL` into a `LikelihoodTerm`
- some edits to `convert` module and exposure of the `convert_from_amptools` method in the main python package
- add gradient calculations at `Amplitude` level
- add `amptools-to-laddu` conversion script to python package
- add python API for likelihood terms and document Rust API
- proof-of-concept for Likelihood terms
- put `Resources` in `Evaluator` behind an `Arc<RwLock<T>>`
- Add `LikelihoodTerm` trait and implement it for `NLL`

### Fixed

- update `example_1.py` to allow running from any directory
- change NLL implementation to properly weight the contribution from MC
- properly handle summations in NLL
- correct type hints
- ensure `extension-module` is used with the `python` feature
- make sure rayon-free build works
- these indices were backwards
- this should correctly reorganize the gradient vectors to all have the same length
- correct some signatures and fix `PyObserver` implementation

### Other

- some stylistic changes to the README
- update README.md to include the first python example
- remove lints
- move kwarg extractor to be near parser
- update `ganesh` to latest version (better default epsilons)
- move parsing of minimizer options to a dedicated function to reduce code duplication
- add sample size specification
- move Likelihood-related code to new `likelihoods` module
- change benchmark config
- store `Expression`s inside `Evaluator`s to simplify call signatures

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
