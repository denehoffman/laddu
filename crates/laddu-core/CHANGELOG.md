# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0](https://github.com/denehoffman/laddu/compare/laddu-core-v0.2.5...laddu-core-v0.3.0) - 2025-02-21

### Added

- switch the MPI implementation to use safe Rust via a RwLock
- update MPI code to use root-node-agnostic methods
- first pass implementation of MPI interface

### Fixed

- calling get_world before use_mpi causes errors
- correct the open method and counts/displs methods

### Other

- *(vectors)* complete tests for vectors module
- *(vectors)* add more vector test coverage
- update all documentation to include MPI modules
- *(vectors)* use custom type for 3/4-vectors rather than trait impl for nalgebra Vectors

## [0.2.5](https://github.com/denehoffman/laddu/compare/laddu-core-v0.2.4...laddu-core-v0.2.5) - 2025-01-28

### Other

- update Cargo.toml dependencies

## [0.2.4](https://github.com/denehoffman/laddu/compare/laddu-core-v0.2.3...laddu-core-v0.2.4) - 2025-01-27

### Fixed

- move `rayon` feature bounds inside methods to clean up the code and avoid duplication

### Other

- *(data)* fix bootstrap tests by changing seed
- update dependencies and remove `rand` and `rand_chacha`

## [0.2.3](https://github.com/denehoffman/laddu/compare/laddu-core-v0.2.2...laddu-core-v0.2.3) - 2025-01-24

### Added

- add `BinnedGuideTerm` under new `experimental` module
- allow users to add `Dataset`s together to form a new `Dataset`

## [0.1.17](https://github.com/denehoffman/laddu/compare/v0.1.16...v0.1.17) - 2025-01-14

### Added

- move binning functions to `utils`
- add `PiecewiseScalar`, `PiecewiseComplexScalar`, and `PiecewisePolarComplexScalar` amplitudes
- add PyVariable to simplify python functions which need to take generic `Variable`s
- improve error handling by getting rid of `unwrap`s wherever possible

### Fixed

- forgot to remove the error types for data reading
- replace `f64` with `Float` in `generate_bin_edges`
- remove `PyResult` on methods which no longer produce errors
- remove requirement that pickled files end in ".pickle" or ".pkl"

### Other

- revert changes to data loading, it's twice as slow
- fix coverage.yml to hopefully include python test coverage
- fix some python doc links
- change test angle for polar complex scalar
- upload coverage for Python code alongside Rust
- update Python documentation to include information about raised exceptions
- update `arrow` and `parquet` to latest version

## [0.1.16](https://github.com/denehoffman/laddu/compare/v0.1.15...v0.1.16) - 2025-01-05

### Added

- add gradient evaluation to `NLL` and `LikelihoodExpression` in Python API
- add 'threads' argument to pretty much everything in Python API

### Fixed

- add better feature guards so all features can technically be used independently

### Other

- add docs to `MCMCOptions` in non-rayon mode and clean up some extra text

## [0.1.15](https://github.com/denehoffman/laddu/compare/v0.1.14...v0.1.15) - 2024-12-21

### Added

- add MCMC samplers from `ganesh` to `laddu`
- use absolute value of mass and width in Breit-Wigner amplitude
- bump versions on `ganesh`, `pyo3`, and (rust) `numpy` and make appropriate updates for each

### Fixed

- make some small corrections to the python example_1 and test out the built-in autocorrelation observer
- minor modification to allow for MSRV of 1.70

### Other

- update the `Future Plans` section of docs (long overdue)
- update README to include a section about MCMC
- ignore pickle files
- use `cargo-hack` to run checks and tests over all features
- update tests to allow them to work with `f32` feature

## [0.1.14](https://github.com/denehoffman/laddu/compare/v0.1.13...v0.1.14) - 2024-12-06

### Added

- add intermediate `Model` and serialization updates
- add perf commands to justfile

## [0.1.13](https://github.com/denehoffman/laddu/compare/v0.1.12...v0.1.13) - 2024-12-05

### Added

- add evaluate_gradient functions where evaluate is available in python API
- add vector indexing and `Event.get_p4_sum` to python API
- add `__sub__`, `__rsub__`, and `__neg__` implementations to Python `Vector3` and `Vector4`

### Fixed

- correct python method to get energy from 4-vector

### Other

- *(python)* write K-Matrix tests
- *(python)* write Zlm tests
- *(python)* write Ylm tests
- *(python)* write Breit-Wigner tests
- remove unused imports
- *(breit_wigner)* properly test BreitWigner gradient
- *(python)* write common amplitude tests
- *(python)* write amplitude submodule tests
- update common tests
- write python tests for variables
- raise minimum python version to 3.8 to allow for numpy v2 constraint
- we were so close
- I'll just write it myself, what could go wrong
- the alpine package is named `build-base`, not `build-essential`
- remove print statement in test
- add tests for Python data submodule
- switch back to clang and add build-essential
- switch from clang to gcc
- add python3-dev and pkgconf to weird linux builds
- add git and clang to the weird OS target builds
- misnamed wheels
- try globing it
- try using direct wheel path to install
- trying this again, maybe if I specify a local path it won't try to pull `laddu` from PyPI
- use --no-index to try to force pip to install the local wheel
- update maturin.yml to run pytest
- *(python)* add tests for python vector API
- update ruff settings globally

## [0.1.12](https://github.com/denehoffman/laddu/compare/v0.1.11...v0.1.12) - 2024-12-04

### Added

- add basic implementation to read directly from ROOT files
- *(bench)* updated benchmark to run over available parallelism

### Fixed

- correct parallelism to allow for proper codspeed benchmarking
- minor fixes for building without rayon/pyo3

### Other

- Merge pull request [#23](https://github.com/denehoffman/laddu/pull/23) from denehoffman/reading-root
- get rid of `from_momentum` method and replace with methods coming from 3-vectors
- change order of four-vector components and modify operation of `boost`
- bump dependencies

## [0.1.11](https://github.com/denehoffman/laddu/compare/v0.1.10...v0.1.11) - 2024-11-29

### Fixed

- major bug/typo in boost method and tests to check for it

## [0.1.10](https://github.com/denehoffman/laddu/compare/v0.1.9...v0.1.10) - 2024-11-20

### Added

- switch API for acceptance correction to not process genmc till projection
- change the way NLLs are constructed to allow the user to specify a generated dataset

### Fixed

- change `NLL` to always use len(accmc) for `n_mc`

### Other

- use pyproject.toml for doc dependencies
- add copy button to code
- update tutorial page
- add under construction notes
- finish unbinned tutorial
- fix doctests and update example_1 results
- switch argument ordering in `Manager.load`
- reorganize main page and include tutorials
- *(docs)* fix doctest with missing parameter

## [0.1.9](https://github.com/denehoffman/laddu/compare/v0.1.8...v0.1.9) - 2024-11-19

### Added

- add no-op implementations for adding 0 to add-able types
- update type hints with __ropts__ and add magic methods to easily pickle `Status`

### Other

- remove unused references
- *(python)* document `as_dict`

## [0.1.8](https://github.com/denehoffman/laddu/compare/v0.1.7...v0.1.8) - 2024-11-09

### Added

- *(data)* make `Event::get_p4_sum` generic over its argument
- *(variables)* add Mandelstam variables
- *(enums)* add `Channel` enum
- *(enums)* add serde to enums
- *(amplitudes)* add `From` impl for `AmplitudeID` to `Expression` conversion
- *(data)* create `test_dataset` method for testing purposes as well as add `Default` impl to `Event`
- *(enums)* add equality comparison to enums and convert to lowercase before string conversion

### Fixed

- *(enums)* Gottfried-Jackson string conversions were accidentally being redirected to Helicity

### Other

- *(python)* fix equations in Mandelstam docs
- fix some documentation issues
- ignore  in codecov, eventually need to test this on the Python side instead
- *(amplitudes)* add unit tests for `ylm`, `zlm`, `breit_wigner`, and `kmatrix` modules
- *(common)* add unit tests for `common` amplitudes
- *(variables)* add unit tests for `variables` module
- *(amplitudes)* add unit tests for `amplitudes` mod
- *(variables)* use new instead of full struct definition for combined `Variable`s
- *(enums)* add unit tests for converting strings to enums
- *(resources)* add unit tests for  module
- *(data)* add unit tests for  module
- correct docs to reflect some recent changes in how NLLs are calculated

## [0.1.7](https://github.com/denehoffman/laddu/compare/v0.1.6...v0.1.7) - 2024-11-08

### Added

- add `NLL::project_with` to do projecting and isolation in one step
- add `__radd__` implementations wherever `__add__` is implemented

### Other

- bump dependency versions
- manipulate features to allow for MSRV of 1.70.0
- use latest rust version
- update readthedocs config in the hopes that it will properly build laddu
- increase TOC depth
- fix broken link

## [0.1.6](https://github.com/denehoffman/laddu/compare/v0.1.5...v0.1.6) - 2024-11-07

### Added

- add methods to serialize/deserialize fit results
- add gamma factor calculation to 4-momentum
- test documentation
- add stable ABI with minimum python version of 3.7
- add python stub file for vectors

### Fixed

- make sure code works if no pol angle/magnitude are provided
- use the unweighted total number of events and divide data likelihood terms by `n_data`
- correct phase in Zlm
- correct `PolAngle` by normalizing the beam vector
- add amplitude module-level documentation
- correct path to sphinx config
- use incremental builds for maturin development

### Other

- add RTDs documentation badge to README and link to repo in docs
- separate command for rebuilding docs and making docfiles
- finish first pass documenting Python API
- fix typo in K-Matrix Rust docs
- resolve lint warning of `len` without `is_empty`
- more documentation for Python API
- fix data format which said that `eps` vectors have a "p" in their column names
- document`vectors` Python API
- add documentation for `Vector3` in Python API
- docstrings are not exported with `maturin develop`
- add documentation commands to justfile
- add automatic documentation and readthedocs support
- update README with codspeed badge

## [0.1.5](https://github.com/denehoffman/laddu/compare/v0.1.4...v0.1.5) - 2024-10-31

### Added

- remove methods to open data into bins or filtered and replace with method on `Dataset`
- wrap `Event`s inside `Dataset`s in `Arc` to reduce bootstrap copying
- add benchmark for opening datasets
- add method to resample datasets (bootstrapping)

### Other

- switch to Codspeed for benchmarking
- update plot and add output txt file for example_1 and reorganize directory structure
- refactor data loading code into a shared function

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
