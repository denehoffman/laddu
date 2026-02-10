# Changelog

## [0.14.2](https://github.com/denehoffman/laddu/compare/py-laddu-v0.14.1...py-laddu-v0.14.2) (2026-02-10)


### Features

* Add `BinnedGuideTerm` under new `experimental` module ([8d0c626](https://github.com/denehoffman/laddu/commit/8d0c626ffb14980ecf60155cd15eebc7fe94ff13))
* Add `conj` operator to `Amplitude`s and `Expression`s ([62dfe28](https://github.com/denehoffman/laddu/commit/62dfe28c044c41deb9542b8390c35874fe51c691))
* Add `Dataset::weighted_bootstrap` ([9f638fb](https://github.com/denehoffman/laddu/commit/9f638fbd3a1af9b8784ff3a647d73314a18aa717))
* Add `evaluate(Variable)` method to `Dataset` and `Event` ([764bd7f](https://github.com/denehoffman/laddu/commit/764bd7f49df4000700a926d91d9a699586b55737))
* Add `PhaseSpaceFactor` amplitude ([a439b99](https://github.com/denehoffman/laddu/commit/a439b9910ad443369e7686898ed0cdc0214dd834))
* Add `PolPhase` amplitude ([07ebd06](https://github.com/denehoffman/laddu/commit/07ebd06f226eaeb2662dc6af54899cd26c8cc999))
* Add available_parallelism function to python API ([b06d620](https://github.com/denehoffman/laddu/commit/b06d620f9c569ceaeb6bbc991f21af7ff684bf5e))
* Add experimental Regularizer likelihood term ([de4ae47](https://github.com/denehoffman/laddu/commit/de4ae47dc2c0a7c74817a6f319e5b7884caebcfe))
* Add method for boosting an event or a dataset to a given rest frame ([1efab27](https://github.com/denehoffman/laddu/commit/1efab274e9ca7dd2196b163b08cada194fea7fdb))
* Add subtraction, division, and negation operations for all Amplitudes and Expressions ([c9b3b3f](https://github.com/denehoffman/laddu/commit/c9b3b3f0f86fc7457255a19ce0773a51caa6526e))
* Add Swarm methods to Python API and update other algorithm initialization methods ([ddc6813](https://github.com/denehoffman/laddu/commit/ddc68134f10075f9c53d20198febf0d329c9c015))
* Add the ability to name likelihood terms and convenience methods for null and unit likelihood terms, sums, and products ([c5352d7](https://github.com/denehoffman/laddu/commit/c5352d7107d858759ad4403bda522eef7d08492b))
* Add unit-valued `Expression` and define convenience methods for summing and multiplying lists of `Amplitude`s ([c1297eb](https://github.com/denehoffman/laddu/commit/c1297ebc5af6b836f9cabd18f775534f6fc71703))
* Add VariableExpressions to handle Dataset filtering ([3f01968](https://github.com/denehoffman/laddu/commit/3f01968574c6e87705dedca92951252302c09364))
* Allow for the use of `sum(list[Dataset])` in Python code ([631fe49](https://github.com/denehoffman/laddu/commit/631fe4999426b1b2bf84df4a2b6e55fc9484616c))
* Allow users to add `Dataset`s together to form a new `Dataset` ([ef6f80e](https://github.com/denehoffman/laddu/commit/ef6f80e72ac0e49363a45bd2ad034d701f13e969))
* Create example_2, a moment analysis ([7dd7c01](https://github.com/denehoffman/laddu/commit/7dd7c01173ea14433a599589d8f6824e84a261dc))
* First pass implementation of MPI interface ([16d8391](https://github.com/denehoffman/laddu/commit/16d8391cddad9b3db3780c1ffbc7fa7b238fe6f6))
* Improvements to `Dataset` conversions and opening methods ([1264ef2](https://github.com/denehoffman/laddu/commit/1264ef26f5a67694396b22afcc70e0fe71132bd1))
* Make `mpi` a feature in `py-laddu` to allow people to build the python package without it ([ae23bb2](https://github.com/denehoffman/laddu/commit/ae23bb24f77f9a88049c4cf1bb00313954b2cbea))
* Redefine eps-&gt;aux in `Event` definition ([45df457](https://github.com/denehoffman/laddu/commit/45df4578c76c7093ec7ee516c017d6847eb9277b))
* Separate parameter logic into a new struct and unify fixing/freeing/renaming ([834a4e7](https://github.com/denehoffman/laddu/commit/834a4e7cfd68b0eba444bf1788d1aa66ac025580))
* Split `laddu` python package into two, with and without MPI support ([e488b46](https://github.com/denehoffman/laddu/commit/e488b46fda4ec723ab1ef452ffc4a411cdcebf78))
* Switch the MPI implementation to use safe Rust via a RwLock ([58ecc24](https://github.com/denehoffman/laddu/commit/58ecc24ecb29cedb01cadc844d960fb7e9f16e1f))
* Update MPI code to use root-node-agnostic methods ([109846e](https://github.com/denehoffman/laddu/commit/109846efd384f8abed59c6c8a0f9bd1b4137c9b3))


### Bug Fixes

* Add a few more missed pyclasses to the py-laddu exports ([d143fb8](https://github.com/denehoffman/laddu/commit/d143fb89099d9b6878b6cac3f828ca0c80820081))
* Add fallback overloads to several amplitudes ([3559264](https://github.com/denehoffman/laddu/commit/35592640d91f822e63b29c4c5c0eb831419c338e))
* Add mpi feature for laddu-python to py-laddu ([02dc5ed](https://github.com/denehoffman/laddu/commit/02dc5edec9dd9a437d22b828a61fd3819f00c859))
* Add non-MPI failing functions for MPI calls on non-MPI python builds ([6be7e24](https://github.com/denehoffman/laddu/commit/6be7e245e5d5e346fa697e28bd900011088c971c))
* Add type hints to `__doc__` and `__version__` ([0c9ffe2](https://github.com/denehoffman/laddu/commit/0c9ffe2e79cedbe43a02efd37e3326b1609ef50d))
* Change `LikelihoodTerm` naming to happen at registration time ([ca3516d](https://github.com/denehoffman/laddu/commit/ca3516db402d19851ad4137b81cb24684f574243))
* Convert range to list ([63cc88b](https://github.com/denehoffman/laddu/commit/63cc88be95789cb3ca4ed0c626138d332a15be4c))
* Correct arguments to fix open_amptools method ([8c1b4d8](https://github.com/denehoffman/laddu/commit/8c1b4d86ef1119eab8994009f44ee5bc2f053978))
* Correct some issues with conversion scripts ([e7a1099](https://github.com/denehoffman/laddu/commit/e7a1099f39acc25376558226d6e64ce70bb3cc04))
* Correct typo AIES-&gt;AEIS in python type checking files ([88dfbc5](https://github.com/denehoffman/laddu/commit/88dfbc50eb44127497e07e6211defb8c19a8245e))
* Corrected signature in methods that read from AmpTools trees ([d751c37](https://github.com/denehoffman/laddu/commit/d751c37027868b4f2e0273a4ba83bd5fb974d406))
* Corrected signature in methods that read from AmpTools trees ([b93ae60](https://github.com/denehoffman/laddu/commit/b93ae60367e5362727344c8d59ecf6381c223099))
* Fixed python examples and readme paths ([56f35a4](https://github.com/denehoffman/laddu/commit/56f35a41b2d7b72fbc887a1b92485b49e4554f2a))
* Forgot to export MCMC moves ([da899fd](https://github.com/denehoffman/laddu/commit/da899fd8822af8c51f3ef46b90dd0335f8b26e4d))
* Missed AEISMove-&gt;AIESMove and condensed the lib.rs file for py-laddu ([a6e9afc](https://github.com/denehoffman/laddu/commit/a6e9afcd9830ebc7a4d9ff01929ebdd80cf493f3))
* Modify tests and workflows to new structure ([33f456c](https://github.com/denehoffman/laddu/commit/33f456c1bc97442792afd4cd55c19ec5aea88b3e))
* Remove loguru and pandas dependencies and add pyarrow dependency ([8048674](https://github.com/denehoffman/laddu/commit/804867484b827659d675e6cb522c5ffba30be5d4))
* Reorganize package structure ([6020e1d](https://github.com/denehoffman/laddu/commit/6020e1dceb5679a27d279dd381894dcc7beb3671))
* The last commit fixed the typo the wrong way, it is AIES ([bc193d6](https://github.com/denehoffman/laddu/commit/bc193d630b9d149240876d7e614a867207133826))
* Update Python vector names (closes [#57](https://github.com/denehoffman/laddu/issues/57)) ([d0f8ee0](https://github.com/denehoffman/laddu/commit/d0f8ee0129333c4dafb04c2804e41aa16145faec))
* Use proper path in `laddu-mpi` entrypoint ([8919d64](https://github.com/denehoffman/laddu/commit/8919d64b0107493dca45f2952450a56a7ff60f2a))


### Reverts

* Remove weighted_bootstrap ([891926f](https://github.com/denehoffman/laddu/commit/891926fac9035f18f7c33900ac2632ce32d88529))


### Documentation

* Add documentation to new Dataset constructors ([a492383](https://github.com/denehoffman/laddu/commit/a492383f735411821d17b23d5b3197ad6704c74c))
* Correct path of extensions module ([1618e1c](https://github.com/denehoffman/laddu/commit/1618e1c6feffdb1ae9735fe1118c158cdd3b089f))
* Create a binned fit tutorial and example ([e1e9904](https://github.com/denehoffman/laddu/commit/e1e99045740958221fa7c65407d53439348041dd))
* Fix python docs to use "extensions" rather than "likelihoods" ([2fa6e5b](https://github.com/denehoffman/laddu/commit/2fa6e5b4825288b6cf5c2351b1edd25a68e662ad))
* Fix some citations and equations, and add phase_space to the API listing ([49965e2](https://github.com/denehoffman/laddu/commit/49965e2e1d28d28edcbfb3e9fccde7bd9d1ad86f))
* Move all MPI code to `laddu-python` to make sure MPI docs build properly ([3ac8b4c](https://github.com/denehoffman/laddu/commit/3ac8b4cbda0ada70a9d48c6ffa5434170493bf0f))
* **mpi:** Add code-block directive to MPI context manager docs ([a91f93f](https://github.com/denehoffman/laddu/commit/a91f93f1b7cab6768fdf8e2c84bd47762f53beed))
* **python:** Add mpi module ([a81f533](https://github.com/denehoffman/laddu/commit/a81f533775e90474dd0c8eb8d5f3706e9154f944))
* **python:** Add MPI to API listing ([72b08b8](https://github.com/denehoffman/laddu/commit/72b08b841c709f86b1490f634dbbf970effe75e6))
* Update all documentation to include MPI modules ([a44ccff](https://github.com/denehoffman/laddu/commit/a44ccff9701fe0b25b9ea95a7676d2cfa51fd933))
* Update moment analysis tutorial and example ([da93613](https://github.com/denehoffman/laddu/commit/da936131bf1c71fcca339236d0b06fb1139ae2ec))
