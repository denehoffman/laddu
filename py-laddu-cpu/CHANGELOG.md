# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.13.1](https://github.com/denehoffman/laddu/compare/py-laddu-cpu-v0.13.0...py-laddu-cpu-v0.13.1) - 2026-01-06

### Added

- separate parameter logic into a new struct and unify fixing/freeing/renaming

## [0.12.2](https://github.com/denehoffman/laddu/compare/py-laddu-cpu-v0.12.1...py-laddu-cpu-v0.12.2) - 2026-01-05

### Other

- Development ([#94](https://github.com/denehoffman/laddu/pull/94))

## [0.12.0](https://github.com/denehoffman/laddu/releases/tag/py-laddu-cpu-v0.12.0) - 2025-12-17

### Added

- add intermediate `Model` and serialization updates
- change the way NLLs are constructed to allow the user to specify a generated dataset
- add `LadduError` struct and work in proper error forwarding for reading data and registering `Amplitude`s
- use `AsRef` generics to allow more versatile `Variable` construction
- first commit

### Fixed

- fixed python examples and readme paths
- modify tests and workflows to new structure

### Other

- release ([#91](https://github.com/denehoffman/laddu/pull/91))
- New file format ([#90](https://github.com/denehoffman/laddu/pull/90))
- update moment analysis tutorial and example
- finalize conversion of eps->aux in data formatting
- update all documentation to include MPI modules
- update the `Future Plans` section of docs (long overdue)
- update README to include a section about MCMC
- correct docs to reflect some recent changes in how NLLs are calculated
- fix broken link
- add RTDs documentation badge to README and link to repo in docs
- fix data format which said that `eps` vectors have a "p" in their column names
- update README with codspeed badge
- update plot and add output txt file for example_1 and reorganize directory structure
- some stylistic changes to the README
- update README.md to include the first python example
- update minimization example in README.md
- change logo to wordmark

## [0.11.0](https://github.com/denehoffman/laddu/releases/tag/py-laddu-cpu-v0.11.0) - 2025-12-17

### Added

- add intermediate `Model` and serialization updates
- change the way NLLs are constructed to allow the user to specify a generated dataset
- add `LadduError` struct and work in proper error forwarding for reading data and registering `Amplitude`s
- use `AsRef` generics to allow more versatile `Variable` construction
- first commit

### Fixed

- fixed python examples and readme paths
- modify tests and workflows to new structure

### Other

- New file format ([#90](https://github.com/denehoffman/laddu/pull/90))
- update moment analysis tutorial and example
- finalize conversion of eps->aux in data formatting
- update all documentation to include MPI modules
- update the `Future Plans` section of docs (long overdue)
- update README to include a section about MCMC
- correct docs to reflect some recent changes in how NLLs are calculated
- fix broken link
- add RTDs documentation badge to README and link to repo in docs
- fix data format which said that `eps` vectors have a "p" in their column names
- update README with codspeed badge
- update plot and add output txt file for example_1 and reorganize directory structure
- some stylistic changes to the README
- update README.md to include the first python example
- update minimization example in README.md
- change logo to wordmark

## [0.9.4](https://github.com/denehoffman/laddu/compare/py-laddu-v0.9.3...py-laddu-v0.9.4) - 2025-10-10

### Other

- Stochastic NLLs and `ganesh` update ([#87](https://github.com/denehoffman/laddu/pull/87))

## [0.9.3](https://github.com/denehoffman/laddu/compare/py-laddu-v0.9.2...py-laddu-v0.9.3) - 2025-08-08

### Added

- add VariableExpressions to handle Dataset filtering

### Other

- fix rounding error in filter test

## [0.9.2](https://github.com/denehoffman/laddu/compare/py-laddu-v0.9.1...py-laddu-v0.9.2) - 2025-07-23

### Fixed

- add more precision to covariance matrices to ensure positive definiteness

## [0.9.1](https://github.com/denehoffman/laddu/compare/py-laddu-v0.9.0...py-laddu-v0.9.1) - 2025-07-23

### Other

- K-Matrix Covariance ([#81](https://github.com/denehoffman/laddu/pull/81))

## [0.8.1](https://github.com/denehoffman/laddu/compare/py-laddu-v0.8.0...py-laddu-v0.8.1) - 2025-06-20

### Added

- create example_2, a moment analysis
- add `conj` operator to `Amplitude`s and `Expression`s
- add `PolPhase` amplitude
- add subtraction, division, and negation operations for all Amplitudes and Expressions

### Other

- update moment analysis tutorial and example
- update dependencies in example_1
- fix printing test

## [0.8.0](https://github.com/denehoffman/laddu/compare/py-laddu-v0.7.1...py-laddu-v0.8.0) - 2025-06-17

### Added

- add `evaluate(Variable)` method to `Dataset` and `Event`

### Other

- update test to get rid of `Arc<Dataset>` structure
- [**breaking**] change `Variable` trait methods to operate on `&Dataset` rather than `&Arc<Dataset>`

## [0.7.1](https://github.com/denehoffman/laddu/compare/py-laddu-v0.7.0...py-laddu-v0.7.1) - 2025-05-30

### Added

- add `Dataset::weighted_bootstrap`

### Fixed

- correct for out-of-bounds errors in MPI bootstrap

### Other

- remove weighted_bootstrap

## [0.6.4](https://github.com/denehoffman/laddu/compare/py-laddu-v0.6.3...py-laddu-v0.6.4) - 2025-05-20

### Added

- improvements to `Dataset` conversions and opening methods
- add method for opening a dataset boosted to the rest frame of the given p4 indices

### Fixed

- update Python vector names (closes #57)

### Other

- add documentation to new Dataset constructors
- *(data)* fix boost tests

## [0.6.2](https://github.com/denehoffman/laddu/compare/py-laddu-v0.6.1...py-laddu-v0.6.2) - 2025-05-16

### Added

- add method for boosting an event or a dataset to a given rest frame

## [0.6.1](https://github.com/denehoffman/laddu/compare/py-laddu-v0.6.0...py-laddu-v0.6.1) - 2025-04-25

### Added

- add available_parallelism function to python API

## [0.5.3](https://github.com/denehoffman/laddu/compare/py-laddu-v0.5.2...py-laddu-v0.5.3) - 2025-04-11

### Added

- add Swarm methods to Python API and update other algorithm initialization methods
- add python versions of Point, Particle, SwarmObserver, and Swarm from ganesh
- change swarm repr if the swarm is uninitialized to not confuse people
- bump MSRV (for bincode) and bump all dependency versions
- restructure the minimizer/mcmc methods to no longer take kwargs
- update `ganesh` version and add Global move to ESS

### Fixed

- add a few more missed pyclasses to the py-laddu exports
- missed AEISMove->AIESMove and condensed the lib.rs file for py-laddu
- forgot to export MCMC moves
- the last commit fixed the typo the wrong way, it is AIES
- correct typo AIES->AEIS in python type checking files
- remove  from the rayon-free  calls for  and
- corrected typo where the `VerboseMCMCObserver` implemented `SwarmObserver<()>` rather than the `VerboseSwarmObserver`
- move some imports under the python feature flag

### Other

- complete compatibility with newest version of bincode, remove unused dependencies and features across all crates
- add a todo

## [0.5.2](https://github.com/denehoffman/laddu/compare/py-laddu-v0.5.1...py-laddu-v0.5.2) - 2025-04-04

### Added

- add experimental Regularizer likelihood term
- update ganesh, numpy, and pyo3

### Fixed

- more fixes for newest ganesh version
- missed a changed path in some ganesh code hidden behind a feature gate

### Other

- fix some citations and equations, and add phase_space to the API listing
- Merge pull request #65 from denehoffman/quality-of-life

## [0.5.1](https://github.com/denehoffman/laddu/compare/py-laddu-v0.5.0...py-laddu-v0.5.1) - 2025-03-16

### Fixed

- change unwrap to print error and panic
- unwrap call_method so that it reports the stack trace if the method fails

## [0.4.2](https://github.com/denehoffman/laddu/compare/py-laddu-v0.4.1...py-laddu-v0.4.2) - 2025-03-13

### Added

- add unit-valued `Expression` and define convenience methods for summing and multiplying lists of `Amplitude`s
- add the ability to name likelihood terms and convenience methods for null and unit likelihood terms, sums, and products
- display the AmplitudeID's name and ID together
- add `Debug` and `Display` to every `Variable` (and require them for new ones)

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

## [0.4.1](https://github.com/denehoffman/laddu/compare/py-laddu-v0.4.0...py-laddu-v0.4.1) - 2025-03-04

### Added

- add `PhaseSpaceFactor` amplitude

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
