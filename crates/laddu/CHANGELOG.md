# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.9.2](https://github.com/denehoffman/laddu/compare/laddu-v0.9.1...laddu-v0.9.2) - 2025-07-23

### Fixed

- add more precision to covariance matrices to ensure positive definiteness

## [0.9.1](https://github.com/denehoffman/laddu/compare/laddu-v0.9.0...laddu-v0.9.1) - 2025-07-23

### Other

- K-Matrix Covariance ([#81](https://github.com/denehoffman/laddu/pull/81))

## [0.8.1](https://github.com/denehoffman/laddu/compare/laddu-v0.8.0...laddu-v0.8.1) - 2025-06-20

### Added

- add `conj` operator to `Amplitude`s and `Expression`s
- add subtraction, division, and negation operations for all Amplitudes and Expressions
- add `PolPhase` amplitude

### Other

- update moment analysis tutorial and example

## [0.8.0](https://github.com/denehoffman/laddu/compare/laddu-v0.7.1...laddu-v0.8.0) - 2025-06-17

### Added

- add `evaluate(Variable)` method to `Dataset` and `Event`

### Other

- update test to get rid of `Arc<Dataset>` structure
- [**breaking**] change `Variable` trait methods to operate on `&Dataset` rather than `&Arc<Dataset>`

## [0.7.1](https://github.com/denehoffman/laddu/compare/laddu-v0.7.0...laddu-v0.7.1) - 2025-05-30

### Added

- add `Dataset::weighted_bootstrap`

### Fixed

- correct for out-of-bounds errors in MPI bootstrap

### Other

- remove weighted_bootstrap

## [0.6.3](https://github.com/denehoffman/laddu/compare/laddu-v0.6.2...laddu-v0.6.3) - 2025-05-20

### Added

- add method for opening a dataset boosted to the rest frame of the given p4 indices

### Other

- *(data)* fix boost tests

## [0.6.2](https://github.com/denehoffman/laddu/compare/laddu-v0.6.1...laddu-v0.6.2) - 2025-05-16

### Added

- add method for boosting an event or a dataset to a given rest frame

## [0.6.1](https://github.com/denehoffman/laddu/compare/laddu-v0.6.0...laddu-v0.6.1) - 2025-04-25

### Other

- updated the following local packages: laddu-amplitudes, laddu-extensions

## [0.5.3](https://github.com/denehoffman/laddu/compare/laddu-v0.5.2...laddu-v0.5.3) - 2025-04-11

### Added

- add Swarm methods to Python API and update other algorithm initialization methods
- add python versions of Point, Particle, SwarmObserver, and Swarm from ganesh
- change swarm repr if the swarm is uninitialized to not confuse people
- bump MSRV (for bincode) and bump all dependency versions
- restructure the minimizer/mcmc methods to no longer take kwargs
- update `ganesh` version and add Global move to ESS

### Fixed

- add feature flag to benchmark to allow it to be run with "f32" feature
- remove  from the rayon-free  calls for  and
- corrected typo where the `VerboseMCMCObserver` implemented `SwarmObserver<()>` rather than the `VerboseSwarmObserver`
- move some imports under the python feature flag

### Other

- complete compatibility with newest version of bincode, remove unused dependencies and features across all crates
- add a todo

## [0.5.2](https://github.com/denehoffman/laddu/compare/laddu-v0.5.1...laddu-v0.5.2) - 2025-04-04

### Added

- add experimental Regularizer likelihood term
- update ganesh, numpy, and pyo3

### Fixed

- more fixes for newest ganesh version
- missed a changed path in some ganesh code hidden behind a feature gate

### Other

- fix some citations and equations, and add phase_space to the API listing
- Merge pull request #65 from denehoffman/quality-of-life

## [0.5.1](https://github.com/denehoffman/laddu/compare/laddu-v0.5.0...laddu-v0.5.1) - 2025-03-16

### Fixed

- change unwrap to print error and panic
- unwrap call_method so that it reports the stack trace if the method fails

## [0.4.2](https://github.com/denehoffman/laddu/compare/laddu-v0.4.1...laddu-v0.4.2) - 2025-03-13

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

## [0.4.1](https://github.com/denehoffman/laddu/compare/laddu-v0.4.0...laddu-v0.4.1) - 2025-03-04

### Added

- add `PhaseSpaceFactor` amplitude

## [0.4.0](https://github.com/denehoffman/laddu/compare/laddu-v0.3.0...laddu-v0.3.1) - 2025-02-28

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
- _(vectors)_ use custom type for 3/4-vectors rather than trait impl for nalgebra Vectors
- update benchmark to only run on powers of 2 threads
- _(vectors)_ complete tests for vectors module
- _(vectors)_ add more vector test coverage
- _(ganesh_ext)_ documenting a few missed functions
- use elided lifetimes

## [0.2.6](https://github.com/denehoffman/laddu/compare/laddu-v0.2.5...laddu-v0.2.6) - 2025-01-28

### Added

- bump `ganesh` to add "skip_hessian" minimization option to skip calculation of Hessian matrix

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
- _(data)_ fix bootstrap tests by changing seed

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
