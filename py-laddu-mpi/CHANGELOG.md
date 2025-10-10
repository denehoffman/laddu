# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.9.4](https://github.com/denehoffman/laddu/compare/py-laddu-mpi-v0.9.3...py-laddu-mpi-v0.9.4) - 2025-10-10

### Other

- Stochastic NLLs and `ganesh` update ([#87](https://github.com/denehoffman/laddu/pull/87))

## [0.9.3](https://github.com/denehoffman/laddu/compare/py-laddu-mpi-v0.9.2...py-laddu-mpi-v0.9.3) - 2025-08-08

### Added

- add VariableExpressions to handle Dataset filtering

### Other

- fix rounding error in filter test

## [0.9.2](https://github.com/denehoffman/laddu/compare/py-laddu-mpi-v0.9.1...py-laddu-mpi-v0.9.2) - 2025-07-23

### Fixed

- add more precision to covariance matrices to ensure positive definiteness

## [0.9.1](https://github.com/denehoffman/laddu/compare/py-laddu-mpi-v0.9.0...py-laddu-mpi-v0.9.1) - 2025-07-23

### Other

- K-Matrix Covariance ([#81](https://github.com/denehoffman/laddu/pull/81))

## [0.8.1](https://github.com/denehoffman/laddu/compare/py-laddu-mpi-v0.8.0...py-laddu-mpi-v0.8.1) - 2025-06-20

### Added

- add `conj` operator to `Amplitude`s and `Expression`s
- add subtraction, division, and negation operations for all Amplitudes and Expressions
- add `PolPhase` amplitude
- create example_2, a moment analysis

### Other

- update moment analysis tutorial and example
- update dependencies in example_1
- fix printing test

## [0.8.0](https://github.com/denehoffman/laddu/compare/py-laddu-mpi-v0.7.1...py-laddu-mpi-v0.8.0) - 2025-06-17

### Added

- add `evaluate(Variable)` method to `Dataset` and `Event`

### Other

- update test to get rid of `Arc<Dataset>` structure
- [**breaking**] change `Variable` trait methods to operate on `&Dataset` rather than `&Arc<Dataset>`

## [0.7.1](https://github.com/denehoffman/laddu/compare/py-laddu-mpi-v0.7.0...py-laddu-mpi-v0.7.1) - 2025-05-30

### Added

- add `Dataset::weighted_bootstrap`

### Fixed

- correct for out-of-bounds errors in MPI bootstrap

### Other

- remove weighted_bootstrap

## [0.6.4](https://github.com/denehoffman/laddu/compare/py-laddu-mpi-v0.6.3...py-laddu-mpi-v0.6.4) - 2025-05-20

### Added

- improvements to `Dataset` conversions and opening methods
- add method for opening a dataset boosted to the rest frame of the given p4 indices

### Fixed

- update Python vector names (closes #57)

### Other

- *(data)* fix boost tests
- add documentation to new Dataset constructors

## [0.6.2](https://github.com/denehoffman/laddu/compare/py-laddu-mpi-v0.6.1...py-laddu-mpi-v0.6.2) - 2025-05-16

### Added

- add method for boosting an event or a dataset to a given rest frame

## [0.6.1](https://github.com/denehoffman/laddu/compare/py-laddu-mpi-v0.6.0...py-laddu-mpi-v0.6.1) - 2025-04-25

### Added

- add available_parallelism function to python API

## [0.5.3](https://github.com/denehoffman/laddu/compare/py-laddu-mpi-v0.5.2...py-laddu-mpi-v0.5.3) - 2025-04-11

### Added

- add Swarm methods to Python API and update other algorithm initialization methods
- add python versions of Point, Particle, SwarmObserver, and Swarm from ganesh
- change swarm repr if the swarm is uninitialized to not confuse people
- bump MSRV (for bincode) and bump all dependency versions
- restructure the minimizer/mcmc methods to no longer take kwargs
- update `ganesh` version and add Global move to ESS

### Fixed

- remove  from the rayon-free  calls for  and
- corrected typo where the `VerboseMCMCObserver` implemented `SwarmObserver<()>` rather than the `VerboseSwarmObserver`
- move some imports under the python feature flag
- add a few more missed pyclasses to the py-laddu exports
- missed AEISMove->AIESMove and condensed the lib.rs file for py-laddu
- forgot to export MCMC moves
- the last commit fixed the typo the wrong way, it is AIES
- correct typo AIES->AEIS in python type checking files

### Other

- complete compatibility with newest version of bincode, remove unused dependencies and features across all crates
- add a todo

## [0.5.2](https://github.com/denehoffman/laddu/compare/py-laddu-mpi-v0.5.1...py-laddu-mpi-v0.5.2) - 2025-04-04

### Added

- add experimental Regularizer likelihood term
- update ganesh, numpy, and pyo3

### Fixed

- more fixes for newest ganesh version
- missed a changed path in some ganesh code hidden behind a feature gate

### Other

- update Cargo.toml dependencies
- fix some citations and equations, and add phase_space to the API listing
- Merge pull request #65 from denehoffman/quality-of-life

## [0.5.1](https://github.com/denehoffman/laddu/compare/py-laddu-mpi-v0.5.0...py-laddu-mpi-v0.5.1) - 2025-03-16

### Fixed

- change unwrap to print error and panic
- unwrap call_method so that it reports the stack trace if the method fails

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
