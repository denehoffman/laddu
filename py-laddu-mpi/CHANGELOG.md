# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.2](https://github.com/denehoffman/laddu/compare/py-laddu-mpi-v0.4.1...py-laddu-mpi-v0.4.2) - 2025-03-13

### Added

- display the AmplitudeID's name and ID together
- add unit-valued `Expression` and define convenience methods for summing and multiplying lists of `Amplitude`s
- add `Debug` and `Display` to every `Variable` (and require them for new ones)
- add the ability to name likelihood terms and convenience methods for null and unit likelihood terms, sums, and products

### Fixed

- update GradientValues in non-default feature branches
- correct gradients of zero and one by adding the number of parameters into `GradientValues`
- update GradientValues in non-default feature branches (missed one)
- improve summation and product methods to only return a Zero or One if the list is empty
- change `LikelihoodTerm` naming to happen at registration time
- add python feature gate to likelihood-related methods

### Other

- *(amplitudes)* expand gradient test to cover more complex cases and add tests for zeros, ones, sums and products
- *(amplitudes)* add test for printing expressions
- *(variables)* add tests for `Variable` Display impls
- *(data)* fix tests by implementing Debug/Display for testing variable
- ignore excessive precision warnings
- *(likelihoods)* fix typo in `NLL` documentation

## [0.4.1](https://github.com/denehoffman/laddu/compare/py-laddu-mpi-v0.4.0...py-laddu-mpi-v0.4.1) - 2025-03-04

### Added

- add `PhaseSpaceFactor` amplitude

## [0.4.0](https://github.com/denehoffman/laddu/releases/tag/py-laddu-mpi-v0.3.0) - 2025-02-28

### Fixed

- reorganize package structure
