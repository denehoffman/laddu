# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.1](https://github.com/denehoffman/laddu/compare/laddu-extensions-v0.3.0...laddu-extensions-v0.3.1) - 2025-02-28

### Other

- fix citation formatting

## [0.3.0](https://github.com/denehoffman/laddu/compare/laddu-extensions-v0.2.6...laddu-extensions-v0.3.0) - 2025-02-21

### Added

- update MPI code to use root-node-agnostic methods
- first pass implementation of MPI interface

### Other

- *(ganesh_ext)* documenting a few missed functions
- update all documentation to include MPI modules
- add some clippy lints and clean up some unused imports and redundant code
- use elided lifetimes

## [0.2.6](https://github.com/denehoffman/laddu/compare/laddu-extensions-v0.2.5...laddu-extensions-v0.2.6) - 2025-01-28

### Added

- bump `ganesh`  to add "skip_hessian" minimization option to skip calculation of Hessian matrix

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
