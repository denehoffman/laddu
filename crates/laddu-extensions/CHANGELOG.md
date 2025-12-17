# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.12.0](https://github.com/denehoffman/laddu/compare/laddu-extensions-v0.11.0...laddu-extensions-v0.12.0) - 2025-12-17

### Other

- New file format ([#90](https://github.com/denehoffman/laddu/pull/90))

## [0.10.0](https://github.com/denehoffman/laddu/compare/laddu-extensions-v0.9.1...laddu-extensions-v0.10.0) - 2025-10-10

### Other

- Stochastic NLLs and `ganesh` update ([#87](https://github.com/denehoffman/laddu/pull/87))

## [0.9.1](https://github.com/denehoffman/laddu/compare/laddu-extensions-v0.9.0...laddu-extensions-v0.9.1) - 2025-08-08

### Other

- updated the following local packages: laddu-core, laddu-python

## [0.8.1](https://github.com/denehoffman/laddu/compare/laddu-extensions-v0.8.0...laddu-extensions-v0.8.1) - 2025-06-20

### Other

- updated the following local packages: laddu-core, laddu-python

## [0.7.2](https://github.com/denehoffman/laddu/compare/laddu-extensions-v0.7.1...laddu-extensions-v0.7.2) - 2025-06-17

### Other

- updated the following local packages: laddu-core, laddu-python

## [0.7.1](https://github.com/denehoffman/laddu/compare/laddu-extensions-v0.7.0...laddu-extensions-v0.7.1) - 2025-05-30

### Other

- updated the following local packages: laddu-core, laddu-python

## [0.6.3](https://github.com/denehoffman/laddu/compare/laddu-extensions-v0.6.2...laddu-extensions-v0.6.3) - 2025-05-20

### Other

- updated the following local packages: laddu-core, laddu-python

## [0.6.2](https://github.com/denehoffman/laddu/compare/laddu-extensions-v0.6.1...laddu-extensions-v0.6.2) - 2025-05-16

### Other

- updated the following local packages: laddu-core, laddu-python

## [0.6.1](https://github.com/denehoffman/laddu/compare/laddu-extensions-v0.6.0...laddu-extensions-v0.6.1) - 2025-04-25

### Other

- updated the following local packages: laddu-python

## [0.6.0](https://github.com/denehoffman/laddu/compare/laddu-extensions-v0.5.2...laddu-extensions-v0.6.0) - 2025-04-11

### Added

- change swarm repr if the swarm is uninitialized to not confuse people
- bump MSRV (for bincode) and bump all dependency versions
- add Swarm methods to Python API and update other algorithm initialization methods
- restructure the minimizer/mcmc methods to no longer take kwargs
- add python versions of Point, Particle, SwarmObserver, and Swarm from ganesh
- update `ganesh` version and add Global move to ESS

### Fixed

- remove  from the rayon-free  calls for  and
- corrected typo where the `VerboseMCMCObserver` implemented `SwarmObserver<()>` rather than the `VerboseSwarmObserver`
- move some imports under the python feature flag

### Other

- complete compatibility with newest version of bincode, remove unused dependencies and features across all crates
- add a todo

## [0.5.2](https://github.com/denehoffman/laddu/compare/laddu-extensions-v0.5.1...laddu-extensions-v0.5.2) - 2025-04-04

### Added

- add experimental Regularizer likelihood term
- update ganesh, numpy, and pyo3

### Fixed

- missed a changed path in some ganesh code hidden behind a feature gate
- more fixes for newest ganesh version

### Other

- Merge pull request #65 from denehoffman/quality-of-life
- fix some citations and equations, and add phase_space to the API listing

## [0.5.1](https://github.com/denehoffman/laddu/compare/laddu-extensions-v0.5.0...laddu-extensions-v0.5.1) - 2025-03-16

### Fixed

- change unwrap to print error and panic
- unwrap call_method so that it reports the stack trace if the method fails

## [0.5.0](https://github.com/denehoffman/laddu/compare/laddu-extensions-v0.4.1...laddu-extensions-v0.5.0) - 2025-03-13

### Added

- add the ability to name likelihood terms and convenience methods for null and unit likelihood terms, sums, and products

### Fixed

- update GradientValues in non-default feature branches (missed one)
- update GradientValues in non-default feature branches
- correct gradients of zero and one by adding the number of parameters into `GradientValues`
- improve summation and product methods to only return a Zero or One if the list is empty
- change `LikelihoodTerm` naming to happen at registration time
- add python feature gate to likelihood-related methods

### Other

- *(likelihoods)* fix typo in `NLL` documentation

## [0.4.1](https://github.com/denehoffman/laddu/compare/laddu-extensions-v0.4.0...laddu-extensions-v0.4.1) - 2025-03-04

### Other

- updated the following local packages: laddu-python

## [0.4.0](https://github.com/denehoffman/laddu/compare/laddu-extensions-v0.3.0...laddu-extensions-v0.3.1) - 2025-02-28

### Other

- fix citation formatting

## [0.3.0](https://github.com/denehoffman/laddu/compare/laddu-extensions-v0.2.6...laddu-extensions-v0.3.0) - 2025-02-21

### Added

- update MPI code to use root-node-agnostic methods
- first pass implementation of MPI interface

### Other

- _(ganesh_ext)_ documenting a few missed functions
- update all documentation to include MPI modules
- add some clippy lints and clean up some unused imports and redundant code
- use elided lifetimes

## [0.2.6](https://github.com/denehoffman/laddu/compare/laddu-extensions-v0.2.5...laddu-extensions-v0.2.6) - 2025-01-28

### Added

- bump `ganesh` to add "skip_hessian" minimization option to skip calculation of Hessian matrix

### Fixed

- use proper ownership in setting algorithm error mode
- use correct enum in L-BFGS-B error method

## [0.2.5](https://github.com/denehoffman/laddu/compare/laddu-extensions-v0.2.4...laddu-extensions-v0.2.5) - 2025-01-27

### Other

- updated the following local packages: laddu-core, laddu-python

## [0.2.4](https://github.com/denehoffman/laddu/compare/laddu-extensions-v0.2.3...laddu-extensions-v0.2.4) - 2025-01-26

### Added

- implement custom gradient for `BinnedGuideTerm`
- add `project_gradient` and `project_gradient_with` methods to `NLL`

### Other

- fix some docstring links in `laddu-extensions`

## [0.2.3](https://github.com/denehoffman/laddu/compare/laddu-extensions-v0.2.2...laddu-extensions-v0.2.3) - 2025-01-24

### Added

- add `BinnedGuideTerm` under new `experimental` module
