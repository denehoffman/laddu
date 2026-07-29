:robot: I have created a release *beep* *boop*
---


<details><summary>0.20.1</summary>

## [0.20.1](https://github.com/denehoffman/laddu/compare/v0.20.0...v0.20.1) (2026-07-29)


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * laddu-amplitudes bumped from 0.20.0 to 0.21.0
    * laddu-autodiff bumped from 0.20.0 to 0.21.0
    * laddu-compile bumped from 0.20.0 to 0.21.0
    * laddu-data bumped from 0.20.0 to 0.21.0
    * laddu-expr bumped from 0.20.0 to 0.21.0
    * laddu-fit bumped from 0.20.0 to 0.21.0
    * laddu-generation bumped from 0.20.0 to 0.21.0
    * laddu-kernel bumped from 0.20.0 to 0.21.0
    * laddu-likelihood bumped from 0.20.0 to 0.21.0
    * laddu-physics bumped from 0.20.0 to 0.21.0
    * laddu-runtime bumped from 0.20.0 to 0.21.0
    * laddu-wgpu bumped from 0.20.0 to 0.21.0
</details>

<details><summary>laddu-amplitudes: 0.21.0</summary>

## [0.21.0](https://github.com/denehoffman/laddu/compare/laddu-amplitudes-v0.20.0...laddu-amplitudes-v0.21.0) (2026-07-29)


###   BREAKING CHANGES

* major rewrite of laddu to use expression trees for amplitudes
* replace legacy laddu code with expression-based rewrite
* remove legacy tree after migration
* **amplitudes:** rename scalar kinematics and Breit-Wigner helpers to the expression-oriented API.
* move Breit-Wigner functions to amplitudes
* **expr:** rebuild expression graph pipeline
* replace the old laddu implementation with the new typed-kernel rewrite foundation.
* remove the ClebschGordan amplitude wrapper and unfinished automatic model exports.
* redesign channel topology and generation APIs
* split core amplitude module
* replace amplitude names with tags
* expose canonical parameter maps
* clarify dataset event APIs
* gate mandelstam by reaction topology
* query reaction particles by id
* **common:** add automatic parameter naming methods to common amplitudes
* overhaul parameter fixing/freeing/renaming
* the normalization of standard relativistic Breit-Wigner amplitude is now unity.
* Renames NLL projection APIs in Rust and Python to weights-focused names and removes old method names.

### Features

* Add `PhaseSpaceFactor` amplitude ([a439b99](https://github.com/denehoffman/laddu/commit/a439b9910ad443369e7686898ed0cdc0214dd834))
* Add `PolPhase` amplitude ([07ebd06](https://github.com/denehoffman/laddu/commit/07ebd06f226eaeb2662dc6af54899cd26c8cc999))
* Add channel species and quantum number APIs ([6755359](https://github.com/denehoffman/laddu/commit/6755359af91b0bca370a0f91fa5fbabc2b8fcdd3))
* Add channel two-body coupling enumeration ([169fe20](https://github.com/denehoffman/laddu/commit/169fe2072c208e4d4890392669641227283a009d))
* Add metadata to Parameters and remove ParameterLike wrapper type ([bf38895](https://github.com/denehoffman/laddu/commit/bf3889545a0216f338b02d1b26499abde1a57862))
* Add production vertex reaction API ([f9c0c70](https://github.com/denehoffman/laddu/commit/f9c0c70353a606d76ff2989b1d972f7f209fe856))
* Add selection rules and more complex quantum state handling, plus a few minor renames and API changes ([6922866](https://github.com/denehoffman/laddu/commit/6922866998d1fa0c8114efc3abb862b155a77baa))
* **amplitudes:** Add composable K-matrix amplitudes ([07edf50](https://github.com/denehoffman/laddu/commit/07edf5034a82b9a7d9521ad1cdf54a6ea5af8199))
* **autodiff:** Add forward gradients ([c98dba9](https://github.com/denehoffman/laddu/commit/c98dba948578b4c1d95d98598e5717d7cea671a2))
* Clarify dataset event APIs ([ea99bad](https://github.com/denehoffman/laddu/commit/ea99bad056f5adda8a105aaac46b6d78ba84aa28))
* **common:** Add automatic parameter naming methods to common amplitudes ([dd1f981](https://github.com/denehoffman/laddu/commit/dd1f9813a060640f0ff2c6aaf950a085a046d712))
* Expose canonical parameter maps ([4f54cf3](https://github.com/denehoffman/laddu/commit/4f54cf30d7f053d3ea0ddfab629bd16c565c7c8e))
* **expr:** Rebuild expression graph pipeline ([fa3a3ed](https://github.com/denehoffman/laddu/commit/fa3a3ed71a1b76f118b41813489d17e0a2f82590))
* First draft of parameter with interior mutability and shared references ([fed640b](https://github.com/denehoffman/laddu/commit/fed640b0660d4923a41663c4e6e9dad7bad7a76b))
* First step of reorganization pass, moved laddu-core into smaller submodules and updated cargo fmt settings ([13f7a24](https://github.com/denehoffman/laddu/commit/13f7a24fd2c70adf5221ba36830455ceb49cfaa8))
* Gate mandelstam by reaction topology ([5e88f33](https://github.com/denehoffman/laddu/commit/5e88f33a6a96407fc17184830cc56f8ab8c3369d))
* Large updates to main APIs and backend performance/memory ([a9049ba](https://github.com/denehoffman/laddu/commit/a9049ba51c324a9ca7abaec2395d03c2722c4e9d))
* Major rewrite of laddu to use expression trees for amplitudes ([5774f78](https://github.com/denehoffman/laddu/commit/5774f78b20b7a351222f8bc5e6bba1e154a7566a))
* More reorganization, laddu-extensions no longer exports any PyO3 ([ff2c9b7](https://github.com/denehoffman/laddu/commit/ff2c9b7426c571797e414c7d5313115c16603ef3))
* Overhaul parameter fixing/freeing/renaming ([a58275a](https://github.com/denehoffman/laddu/commit/a58275a0e7c4c194d65edbd8bbdc5d3d5c4bac71))
* **python:** Add native package ecosystem ([20df1fb](https://github.com/denehoffman/laddu/commit/20df1fbad711534e84564040bef5b659c2a034a1))
* Query reaction particles by id ([a7a2938](https://github.com/denehoffman/laddu/commit/a7a29385cfe84cef3198aa71c090b3f3c2c5a86e))
* Redefine eps-&gt;aux in `Event` definition ([45df457](https://github.com/denehoffman/laddu/commit/45df4578c76c7093ec7ee516c017d6847eb9277b))
* Redesign channel topology and generation APIs ([9d3e3a4](https://github.com/denehoffman/laddu/commit/9d3e3a44467a6ac14192e064430b1e4932770714))
* Replace amplitude names with tags ([93b2dde](https://github.com/denehoffman/laddu/commit/93b2dde9ed0f85212d653c95d029ea78b4beebfa))
* Replace legacy laddu code with expression-based rewrite ([1ccc1a3](https://github.com/denehoffman/laddu/commit/1ccc1a3c000adcd7d8e41182bcc8f0b72401b0cc))
* Rework kinematics API to allow for full Helicity/Canonical coupling terms, and other changes and improvements ([d6d49a5](https://github.com/denehoffman/laddu/commit/d6d49a5a66da6119b286c73637951954d77c2efb))
* Start breaking laddu kernel rewrite ([8051e64](https://github.com/denehoffman/laddu/commit/8051e6444bba62877abdb62c2a06fbd002a0334d))
* Update quantum helpers to use typed inputs, remove Wigner3J amplitude, add operation overloads for M, Charge, and Parity and orbital_parity for L ([3d979ca](https://github.com/denehoffman/laddu/commit/3d979cacc4b348a7aaa744ec0dd738e38e4ce446))


### Bug Fixes

* Add more precision to covariance matrices to ensure positive definiteness ([49e43ae](https://github.com/denehoffman/laddu/commit/49e43ae61414448aa4b48584ec4415e0afd31bf6))
* **ci:** Pass pre-push verification ([367c169](https://github.com/denehoffman/laddu/commit/367c16951e8792c81f1203ce6af7ba321b5c0e5d))
* Clear clippy lints, especially Errors/Panics sections in docstrings ([2f998db](https://github.com/denehoffman/laddu/commit/2f998db89faeeed9ed70822c3a1d25081873e1b1))
* Remove production frame selection ([09a548e](https://github.com/denehoffman/laddu/commit/09a548e68f93d047e9ba890e33c488751cd40127))
* Reorganize all crates and provide convenience methods for dealing with parameters ([1751265](https://github.com/denehoffman/laddu/commit/1751265ba5473cad0a92eedbde6286afe60c8b92))


### Miscellaneous Chores

* Remove legacy tree after migration ([cf43319](https://github.com/denehoffman/laddu/commit/cf43319bbee4357e2697857bdb2be69edfd10a83))


### Code Refactoring

* Move Breit-Wigner functions to amplitudes ([e990266](https://github.com/denehoffman/laddu/commit/e990266160ff779928411a73358b6c906b4a6a02))
* Split core amplitude module ([add6e70](https://github.com/denehoffman/laddu/commit/add6e70bbfafe7721ada82457998ec100eaf443f))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * laddu-expr bumped from 0.20.0 to 0.21.0
    * laddu-kernel bumped from 0.20.0 to 0.21.0
    * laddu-physics bumped from 0.20.0 to 0.21.0
  * dev-dependencies
    * laddu-compile bumped from 0.20.0 to 0.21.0
    * laddu-runtime bumped from 0.20.0 to 0.21.0
</details>

<details><summary>laddu-autodiff: 0.21.0</summary>

## [0.21.0](https://github.com/denehoffman/laddu/compare/laddu-autodiff-v0.20.0...laddu-autodiff-v0.21.0) (2026-07-29)


###   BREAKING CHANGES

* major rewrite of laddu to use expression trees for amplitudes
* replace legacy laddu code with expression-based rewrite
* **expr:** lower complex parameters to expressions
* **expr:** rebuild expression graph pipeline
* replace the old laddu implementation with the new typed-kernel rewrite foundation.
* the normalization of standard relativistic Breit-Wigner amplitude is now unity.

### Features

* Add serde support to public API types ([9e0ec82](https://github.com/denehoffman/laddu/commit/9e0ec82936c27ec070587ddf8b4e3ae8d16d7acd))
* **autodiff:** Add backend-neutral gradient IR ([e27f185](https://github.com/denehoffman/laddu/commit/e27f18554a8a3f15d301ffc00754ee2804c4cc85))
* **autodiff:** Add forward gradients ([c98dba9](https://github.com/denehoffman/laddu/commit/c98dba948578b4c1d95d98598e5717d7cea671a2))
* **expr:** Rebuild expression graph pipeline ([fa3a3ed](https://github.com/denehoffman/laddu/commit/fa3a3ed71a1b76f118b41813489d17e0a2f82590))
* Major rewrite of laddu to use expression trees for amplitudes ([5774f78](https://github.com/denehoffman/laddu/commit/5774f78b20b7a351222f8bc5e6bba1e154a7566a))
* **python:** Add native package ecosystem ([20df1fb](https://github.com/denehoffman/laddu/commit/20df1fbad711534e84564040bef5b659c2a034a1))
* Replace legacy laddu code with expression-based rewrite ([1ccc1a3](https://github.com/denehoffman/laddu/commit/1ccc1a3c000adcd7d8e41182bcc8f0b72401b0cc))
* Rework kinematics API to allow for full Helicity/Canonical coupling terms, and other changes and improvements ([d6d49a5](https://github.com/denehoffman/laddu/commit/d6d49a5a66da6119b286c73637951954d77c2efb))
* **runtime:** Add f64 cpu reverse autograd ([75cd3c9](https://github.com/denehoffman/laddu/commit/75cd3c9838f809e37412b18a368cf6578152c545))
* **runtime:** Complete CPU gradient JIT parity ([2dc038a](https://github.com/denehoffman/laddu/commit/2dc038abe4eba9d67f3b5cc75f680075ca8166c4))
* **runtime:** Unify gradient kernel execution ([16a4fc0](https://github.com/denehoffman/laddu/commit/16a4fc08598839198713d9e9002d6ee8e583e9a0))
* Start breaking laddu kernel rewrite ([8051e64](https://github.com/denehoffman/laddu/commit/8051e6444bba62877abdb62c2a06fbd002a0334d))


### Bug Fixes

* Change ExprId to u64 to remove some expects and clear all clippy lints ([985c8b8](https://github.com/denehoffman/laddu/commit/985c8b840b56a7df644ca4c8da151921664d62de))
* Clear clippy lints, especially Errors/Panics sections in docstrings ([2f998db](https://github.com/denehoffman/laddu/commit/2f998db89faeeed9ed70822c3a1d25081873e1b1))
* Reorganize all crates and provide convenience methods for dealing with parameters ([1751265](https://github.com/denehoffman/laddu/commit/1751265ba5473cad0a92eedbde6286afe60c8b92))


### Code Refactoring

* **expr:** Lower complex parameters to expressions ([5c528d2](https://github.com/denehoffman/laddu/commit/5c528d28aeb1afeb6678e76a8c329ad9f8efa453))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * laddu-compile bumped from 0.20.0 to 0.21.0
    * laddu-expr bumped from 0.20.0 to 0.21.0
    * laddu-kernel bumped from 0.20.0 to 0.21.0
</details>

<details><summary>laddu-compile: 0.21.0</summary>

## [0.21.0](https://github.com/denehoffman/laddu/compare/laddu-compile-v0.20.0...laddu-compile-v0.21.0) (2026-07-29)


###   BREAKING CHANGES

* major rewrite of laddu to use expression trees for amplitudes
* replace legacy laddu code with expression-based rewrite
* add metadata-aware fitting and closure projections
* **runtime:** integrate WGPU likelihood execution
* **runtime:** add compiled reduction plans
* **expr:** lower complex parameters to expressions
* **amplitudes:** rename scalar kinematics and Breit-Wigner helpers to the expression-oriented API.
* ExprGraph Display now emits expression syntax; use display_tree() for the previous labeled tree output.
* **compile:** normalize n-ary algebra
* **compile:** the default optimizer now runs canonicalization and rewrite passes until graph shape converges, further changing optimized graph shape and operation order.
* **compile:** the default optimizer now reassociates canonical Add/Mul trees and merges exp products, further changing optimized graph shape and floating-point operation order.
* **compile:** expression graphs now include first-class Complex nodes and the default compile pipeline performs aggressive canonicalization/CSE.
* **expr:** rebuild expression graph pipeline
* replace the old laddu implementation with the new typed-kernel rewrite foundation.
* the normalization of standard relativistic Breit-Wigner amplitude is now unity.

### Features

* Add channel kinematics and p4 event expressions ([d9d4c81](https://github.com/denehoffman/laddu/commit/d9d4c813cdc434c3097ed085a679e82afa3234f9))
* Add configurable graph visualization ([d1d947e](https://github.com/denehoffman/laddu/commit/d1d947ec7c412930eab3005f5873296c68cd3454))
* Add extended NLL and parameter recompilation ([4055c16](https://github.com/denehoffman/laddu/commit/4055c16d6cfa6ba6cee740174ceb589d65aa976f))
* Add lazy dataset queries and projections ([fbfde99](https://github.com/denehoffman/laddu/commit/fbfde995b7903e04833b1633cffb9f458d8f6f1d))
* Add metadata-aware fitting and closure projections ([f6d7857](https://github.com/denehoffman/laddu/commit/f6d78577b2c2f6cb6f6d707e58be03e8a9327eca))
* **amplitudes:** Add composable K-matrix amplitudes ([07edf50](https://github.com/denehoffman/laddu/commit/07edf5034a82b9a7d9521ad1cdf54a6ea5af8199))
* **compile:** Add canonical CSE and complex IR ([8c660ca](https://github.com/denehoffman/laddu/commit/8c660ca657c3e52e97e31bcbc6f4caf45740d20c))
* **compile:** Add optimization cost model ([d0454db](https://github.com/denehoffman/laddu/commit/d0454dba220c28bea3e5810ef24fbca0a62c2f96))
* **compile:** Add scalar simplification passes ([17f0586](https://github.com/denehoffman/laddu/commit/17f05868701822502f74e2b4bfaa42f141bae6a0))
* **compile:** Choose factoring rewrites by cost ([e67797b](https://github.com/denehoffman/laddu/commit/e67797b6e2c0c015f121ba57db7bfda1bbac6aa8))
* **compile:** Combine like terms ([384e053](https://github.com/denehoffman/laddu/commit/384e0539290c10d2950e5cea1b6738f372731b04))
* **compile:** Combine same-power product factors ([ef7bc93](https://github.com/denehoffman/laddu/commit/ef7bc934b2545338601ce2711ca136c6fc820134))
* **compile:** Gate norm-sqr expansion by optimizer cost ([ff4b774](https://github.com/denehoffman/laddu/commit/ff4b774cc3df10aae54787ab1c2353b68d145e1d))
* **compile:** Iterate optimizer to fixed point ([3a2b38a](https://github.com/denehoffman/laddu/commit/3a2b38a3971d014967cae59c4a27671f522745e8))
* **compile:** Merge exponential products ([5e91de5](https://github.com/denehoffman/laddu/commit/5e91de5ad9d2a0df6a293d7e6e14bed9bafcf144))
* **compile:** Normalize n-ary algebra ([8bdbd74](https://github.com/denehoffman/laddu/commit/8bdbd74718b72dafd7685ec25ef4b2ec2b932346))
* **compile:** Scalarize selected aggregate outputs ([678e416](https://github.com/denehoffman/laddu/commit/678e416062cfc241f783d68dac38ef8792973e26))
* **compile:** Simplify trigonometric phase forms ([7a45190](https://github.com/denehoffman/laddu/commit/7a451909e04c1a086438ca435a2f7196a1b21995))
* **expr:** Rebuild expression graph pipeline ([fa3a3ed](https://github.com/denehoffman/laddu/commit/fa3a3ed71a1b76f118b41813489d17e0a2f82590))
* Improve graph optimization and display ([8c0409f](https://github.com/denehoffman/laddu/commit/8c0409f3d20222e91ae066c0520657cb4dbe72a9))
* **kernel:** Add cache materialization ir ([7c8f774](https://github.com/denehoffman/laddu/commit/7c8f774e5ec5ecae6f7009a25069bdc94e255844))
* Major rewrite of laddu to use expression trees for amplitudes ([5774f78](https://github.com/denehoffman/laddu/commit/5774f78b20b7a351222f8bc5e6bba1e154a7566a))
* **python:** Add native package ecosystem ([20df1fb](https://github.com/denehoffman/laddu/commit/20df1fbad711534e84564040bef5b659c2a034a1))
* Replace legacy laddu code with expression-based rewrite ([1ccc1a3](https://github.com/denehoffman/laddu/commit/1ccc1a3c000adcd7d8e41182bcc8f0b72401b0cc))
* Rework kinematics API to allow for full Helicity/Canonical coupling terms, and other changes and improvements ([d6d49a5](https://github.com/denehoffman/laddu/commit/d6d49a5a66da6119b286c73637951954d77c2efb))
* **runtime:** Add compiled reduction plans ([aaff110](https://github.com/denehoffman/laddu/commit/aaff1103a7986f45dff0464a899487e57fb3ecb7))
* **runtime:** Add dataset-resident event caches ([fa52ebc](https://github.com/denehoffman/laddu/commit/fa52ebcdf2c52c544a1f79b2e7fc96a73b1793a3))
* **runtime:** Add typed cache layouts ([d4108ba](https://github.com/denehoffman/laddu/commit/d4108ba3a6646b9d94e21ac852680dfa25a88a50))
* **runtime:** Integrate WGPU likelihood execution ([05673fc](https://github.com/denehoffman/laddu/commit/05673fcee7dc5e44c7bac952e0681d630ca94f11))
* Start breaking laddu kernel rewrite ([8051e64](https://github.com/denehoffman/laddu/commit/8051e6444bba62877abdb62c2a06fbd002a0334d))
* **wgpu:** Add scalar gradient reductions ([9061b6a](https://github.com/denehoffman/laddu/commit/9061b6a3920253add7ee36c37d908501ebeb6253))
* **wgpu:** Support aggregate algebra ([8c540f4](https://github.com/denehoffman/laddu/commit/8c540f469bc9b87ffb564ca3c99c4704e995b47f))


### Bug Fixes

* Change ExprId to u64 to remove some expects and clear all clippy lints ([985c8b8](https://github.com/denehoffman/laddu/commit/985c8b840b56a7df644ca4c8da151921664d62de))
* Clear clippy lints, especially Errors/Panics sections in docstrings ([2f998db](https://github.com/denehoffman/laddu/commit/2f998db89faeeed9ed70822c3a1d25081873e1b1))
* Reorganize all crates and provide convenience methods for dealing with parameters ([1751265](https://github.com/denehoffman/laddu/commit/1751265ba5473cad0a92eedbde6286afe60c8b92))


### Performance Improvements

* **compile:** Cache only event frontier nodes ([546e9de](https://github.com/denehoffman/laddu/commit/546e9de4f3df6d8f72bba6564b94e6fcb3e2274e))


### Code Refactoring

* **expr:** Lower complex parameters to expressions ([5c528d2](https://github.com/denehoffman/laddu/commit/5c528d28aeb1afeb6678e76a8c329ad9f8efa453))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * laddu-expr bumped from 0.20.0 to 0.21.0
    * laddu-kernel bumped from 0.20.0 to 0.21.0
</details>

<details><summary>laddu-data: 0.21.0</summary>

## [0.21.0](https://github.com/denehoffman/laddu/compare/laddu-data-v0.20.0...laddu-data-v0.21.0) (2026-07-29)


###   BREAKING CHANGES

* major rewrite of laddu to use expression trees for amplitudes
* replace legacy laddu code with expression-based rewrite
* make memory budgets first-class
* **physics:** Four-vector constructors, conversions, public fields, and positional arrays now use (E, px, py, pz) order.
* expose direct ganesh fit and generation APIs
* **runtime:** unify dataset execution policies
* **physics:** add expression vector builders
* replace the old laddu implementation with the new typed-kernel rewrite foundation.
* the normalization of standard relativistic Breit-Wigner amplitude is now unity.

### Features

* Add channel kinematics and p4 event expressions ([d9d4c81](https://github.com/denehoffman/laddu/commit/d9d4c813cdc434c3097ed085a679e82afa3234f9))
* Add lazy dataset queries and projections ([fbfde99](https://github.com/denehoffman/laddu/commit/fbfde995b7903e04833b1633cffb9f458d8f6f1d))
* Add serde support to public API types ([9e0ec82](https://github.com/denehoffman/laddu/commit/9e0ec82936c27ec070587ddf8b4e3ae8d16d7acd))
* Expose direct ganesh fit and generation APIs ([4525a01](https://github.com/denehoffman/laddu/commit/4525a01feac3855232de9a2aad7713336d90dd3c))
* Improve public Rust API ergonomics ([86ce01b](https://github.com/denehoffman/laddu/commit/86ce01b6758e4db875aec021e40d9ed635f02199))
* **likelihood:** Add cached normalized intensity fits ([6b90132](https://github.com/denehoffman/laddu/commit/6b90132584c4a3b7964164b20b8884924c948150))
* Major rewrite of laddu to use expression trees for amplitudes ([5774f78](https://github.com/denehoffman/laddu/commit/5774f78b20b7a351222f8bc5e6bba1e154a7566a))
* Make memory budgets first-class ([171e658](https://github.com/denehoffman/laddu/commit/171e658ed8cc1a69330f9a68d488e3061e498a55))
* **physics:** Add expression vector builders ([09c0552](https://github.com/denehoffman/laddu/commit/09c0552be110480d9705d92ce23797a50710c143))
* **physics:** Unify four-vector component order ([ad50eca](https://github.com/denehoffman/laddu/commit/ad50eca17b6a3e992494deb6b2794a8acd6e97dc))
* **python:** Add native package ecosystem ([20df1fb](https://github.com/denehoffman/laddu/commit/20df1fbad711534e84564040bef5b659c2a034a1))
* Replace legacy laddu code with expression-based rewrite ([1ccc1a3](https://github.com/denehoffman/laddu/commit/1ccc1a3c000adcd7d8e41182bcc8f0b72401b0cc))
* Rework kinematics API to allow for full Helicity/Canonical coupling terms, and other changes and improvements ([d6d49a5](https://github.com/denehoffman/laddu/commit/d6d49a5a66da6119b286c73637951954d77c2efb))
* **runtime:** Unify dataset execution policies ([3c85961](https://github.com/denehoffman/laddu/commit/3c85961909dab88f82cb2ff2d37171dc6e2c5408))
* Start breaking laddu kernel rewrite ([8051e64](https://github.com/denehoffman/laddu/commit/8051e6444bba62877abdb62c2a06fbd002a0334d))


### Bug Fixes

* Change ExprId to u64 to remove some expects and clear all clippy lints ([985c8b8](https://github.com/denehoffman/laddu/commit/985c8b840b56a7df644ca4c8da151921664d62de))
* **ci:** Pass pre-push verification ([367c169](https://github.com/denehoffman/laddu/commit/367c16951e8792c81f1203ce6af7ba321b5c0e5d))
* Clear clippy lints, especially Errors/Panics sections in docstrings ([2f998db](https://github.com/denehoffman/laddu/commit/2f998db89faeeed9ed70822c3a1d25081873e1b1))
* Reorganize all crates and provide convenience methods for dealing with parameters ([1751265](https://github.com/denehoffman/laddu/commit/1751265ba5473cad0a92eedbde6286afe60c8b92))
</details>

<details><summary>laddu-expr: 0.21.0</summary>

## [0.21.0](https://github.com/denehoffman/laddu/compare/laddu-expr-v0.20.0...laddu-expr-v0.21.0) (2026-07-29)


###   BREAKING CHANGES

* major rewrite of laddu to use expression trees for amplitudes
* replace legacy laddu code with expression-based rewrite
* make memory budgets first-class
* **physics:** Four-vector constructors, conversions, public fields, and positional arrays now use (E, px, py, pz) order.
* add metadata-aware fitting and closure projections
* **generation:** remove FixedInitialState and InitialStateSampler; initial momentum sources now belong to channel edges and ChannelGenerator::new accepts only a Channel.
* **expr:** lower complex parameters to expressions
* **likelihood:** port K-matrix NLL benchmark
* **amplitudes:** rename scalar kinematics and Breit-Wigner helpers to the expression-oriented API.
* ExprGraph Display now emits expression syntax; use display_tree() for the previous labeled tree output.
* **compile:** normalize n-ary algebra
* **compile:** expression graphs now include first-class Complex nodes and the default compile pipeline performs aggressive canonicalization/CSE.
* **expr:** rebuild expression graph pipeline
* replace the old laddu implementation with the new typed-kernel rewrite foundation.
* the normalization of standard relativistic Breit-Wigner amplitude is now unity.

### Features

* Add channel kinematics and p4 event expressions ([d9d4c81](https://github.com/denehoffman/laddu/commit/d9d4c813cdc434c3097ed085a679e82afa3234f9))
* Add configurable graph visualization ([d1d947e](https://github.com/denehoffman/laddu/commit/d1d947ec7c412930eab3005f5873296c68cd3454))
* Add extended NLL and parameter recompilation ([4055c16](https://github.com/denehoffman/laddu/commit/4055c16d6cfa6ba6cee740174ceb589d65aa976f))
* Add lazy dataset queries and projections ([fbfde99](https://github.com/denehoffman/laddu/commit/fbfde995b7903e04833b1633cffb9f458d8f6f1d))
* Add metadata-aware fitting and closure projections ([f6d7857](https://github.com/denehoffman/laddu/commit/f6d78577b2c2f6cb6f6d707e58be03e8a9327eca))
* **amplitudes:** Add composable K-matrix amplitudes ([07edf50](https://github.com/denehoffman/laddu/commit/07edf5034a82b9a7d9521ad1cdf54a6ea5af8199))
* **compile:** Add canonical CSE and complex IR ([8c660ca](https://github.com/denehoffman/laddu/commit/8c660ca657c3e52e97e31bcbc6f4caf45740d20c))
* **compile:** Normalize n-ary algebra ([8bdbd74](https://github.com/denehoffman/laddu/commit/8bdbd74718b72dafd7685ec25ef4b2ec2b932346))
* **expr:** Rebuild expression graph pipeline ([fa3a3ed](https://github.com/denehoffman/laddu/commit/fa3a3ed71a1b76f118b41813489d17e0a2f82590))
* **fit:** Integrate ganesh optimization and sampling ([79abcca](https://github.com/denehoffman/laddu/commit/79abcca5e12aa734c2616e93bcce6fc52cf3eed2))
* **generation:** Add channel-driven event generation ([bdc0bd2](https://github.com/denehoffman/laddu/commit/bdc0bd212cbc72cee6637d9ddf78f1db59549737))
* Improve graph optimization and display ([8c0409f](https://github.com/denehoffman/laddu/commit/8c0409f3d20222e91ae066c0520657cb4dbe72a9))
* Major rewrite of laddu to use expression trees for amplitudes ([5774f78](https://github.com/denehoffman/laddu/commit/5774f78b20b7a351222f8bc5e6bba1e154a7566a))
* Make memory budgets first-class ([171e658](https://github.com/denehoffman/laddu/commit/171e658ed8cc1a69330f9a68d488e3061e498a55))
* Periodic parameters, objectives, and a between query, as well as organizational changes to prepare for Python API ([b1e004f](https://github.com/denehoffman/laddu/commit/b1e004f1aa0b8c16075bbb1e19c580377ecc3317))
* **physics:** Unify four-vector component order ([ad50eca](https://github.com/denehoffman/laddu/commit/ad50eca17b6a3e992494deb6b2794a8acd6e97dc))
* **python:** Add native package ecosystem ([20df1fb](https://github.com/denehoffman/laddu/commit/20df1fbad711534e84564040bef5b659c2a034a1))
* Replace legacy laddu code with expression-based rewrite ([1ccc1a3](https://github.com/denehoffman/laddu/commit/1ccc1a3c000adcd7d8e41182bcc8f0b72401b0cc))
* Rework kinematics API to allow for full Helicity/Canonical coupling terms, and other changes and improvements ([d6d49a5](https://github.com/denehoffman/laddu/commit/d6d49a5a66da6119b286c73637951954d77c2efb))
* Start breaking laddu kernel rewrite ([8051e64](https://github.com/denehoffman/laddu/commit/8051e6444bba62877abdb62c2a06fbd002a0334d))


### Bug Fixes

* Change ExprId to u64 to remove some expects and clear all clippy lints ([985c8b8](https://github.com/denehoffman/laddu/commit/985c8b840b56a7df644ca4c8da151921664d62de))
* Clear clippy lints, especially Errors/Panics sections in docstrings ([2f998db](https://github.com/denehoffman/laddu/commit/2f998db89faeeed9ed70822c3a1d25081873e1b1))
* Reorganize all crates and provide convenience methods for dealing with parameters ([1751265](https://github.com/denehoffman/laddu/commit/1751265ba5473cad0a92eedbde6286afe60c8b92))


### Performance Improvements

* **compile:** Cache only event frontier nodes ([546e9de](https://github.com/denehoffman/laddu/commit/546e9de4f3df6d8f72bba6564b94e6fcb3e2274e))
* **likelihood:** Port K-matrix NLL benchmark ([05d213d](https://github.com/denehoffman/laddu/commit/05d213d8d10ac73b8311af7a6c13e01395f2334d))


### Code Refactoring

* **expr:** Lower complex parameters to expressions ([5c528d2](https://github.com/denehoffman/laddu/commit/5c528d28aeb1afeb6678e76a8c329ad9f8efa453))
</details>

<details><summary>laddu-fit: 0.21.0</summary>

## [0.21.0](https://github.com/denehoffman/laddu/compare/laddu-fit-v0.20.0...laddu-fit-v0.21.0) (2026-07-29)


###   BREAKING CHANGES

* major rewrite of laddu to use expression trees for amplitudes
* replace legacy laddu code with expression-based rewrite
* expose direct ganesh fit and generation APIs
* add metadata-aware fitting and closure projections
* the normalization of standard relativistic Breit-Wigner amplitude is now unity.

### Features

* Add metadata-aware fitting and closure projections ([f6d7857](https://github.com/denehoffman/laddu/commit/f6d78577b2c2f6cb6f6d707e58be03e8a9327eca))
* Expose direct ganesh fit and generation APIs ([4525a01](https://github.com/denehoffman/laddu/commit/4525a01feac3855232de9a2aad7713336d90dd3c))
* **fit:** Accept named initial values and seed walkers ([582e63a](https://github.com/denehoffman/laddu/commit/582e63a11fd7c95dc091f806016d9e8c7bce3f82))
* **fit:** Add generation closure workflow ([4556eaa](https://github.com/denehoffman/laddu/commit/4556eaa4c358ad05beee3482e218354108a75f63))
* **fit:** Integrate ganesh optimization and sampling ([79abcca](https://github.com/denehoffman/laddu/commit/79abcca5e12aa734c2616e93bcce6fc52cf3eed2))
* Major rewrite of laddu to use expression trees for amplitudes ([5774f78](https://github.com/denehoffman/laddu/commit/5774f78b20b7a351222f8bc5e6bba1e154a7566a))
* **python:** Add native package ecosystem ([20df1fb](https://github.com/denehoffman/laddu/commit/20df1fbad711534e84564040bef5b659c2a034a1))
* Replace legacy laddu code with expression-based rewrite ([1ccc1a3](https://github.com/denehoffman/laddu/commit/1ccc1a3c000adcd7d8e41182bcc8f0b72401b0cc))
* Rework kinematics API to allow for full Helicity/Canonical coupling terms, and other changes and improvements ([d6d49a5](https://github.com/denehoffman/laddu/commit/d6d49a5a66da6119b286c73637951954d77c2efb))


### Bug Fixes

* Clear clippy lints, especially Errors/Panics sections in docstrings ([2f998db](https://github.com/denehoffman/laddu/commit/2f998db89faeeed9ed70822c3a1d25081873e1b1))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * laddu-expr bumped from 0.20.0 to 0.21.0
    * laddu-likelihood bumped from 0.20.0 to 0.21.0
    * laddu-runtime bumped from 0.20.0 to 0.21.0
</details>

<details><summary>laddu-generation: 0.21.0</summary>

## [0.21.0](https://github.com/denehoffman/laddu/compare/laddu-generation-v0.20.0...laddu-generation-v0.21.0) (2026-07-29)


###   BREAKING CHANGES

* major rewrite of laddu to use expression trees for amplitudes
* replace legacy laddu code with expression-based rewrite
* remove legacy tree after migration
* make memory budgets first-class
* **physics:** Four-vector constructors, conversions, public fields, and positional arrays now use (E, px, py, pz) order.
* expose direct ganesh fit and generation APIs
* add metadata-aware fitting and closure projections
* **generation:** UnweightedConfig::new now takes only the requested event count, and max_proposals is Option<usize>; use with_max_proposals to opt into a limit.
* **generation:** remove FixedInitialState and InitialStateSampler; initial momentum sources now belong to channel edges and ChannelGenerator::new accepts only a Channel.
* **generation:** replace dataset generation with sinks
* redesign channel topology and generation APIs
* clarify dataset event APIs
* make dataset access explicit
* redesign generation particle API
* query reaction particles by id
* the normalization of standard relativistic Breit-Wigner amplitude is now unity.

### Features

* Add appendable dataset api ([14d7b51](https://github.com/denehoffman/laddu/commit/14d7b51ffb88f509c226c85707671c89d307fc21))
* Add aux columns to generation and formalize MPI standards for generation ([df80f7c](https://github.com/denehoffman/laddu/commit/df80f7c66f57a6f626459a84d876ee21fc686377))
* Add channel species and quantum number APIs ([6755359](https://github.com/denehoffman/laddu/commit/6755359af91b0bca370a0f91fa5fbabc2b8fcdd3))
* Add deterministic generated batch iterator ([5e7f7e3](https://github.com/denehoffman/laddu/commit/5e7f7e322f8c798b1c0085dd1dde88a55625e609))
* Add envelope estimation to rejection sampling ([054ad6b](https://github.com/denehoffman/laddu/commit/054ad6bfcd4513acd40195358f9a112536f15f1c))
* Add expression rejection sampling ([1b57229](https://github.com/denehoffman/laddu/commit/1b57229073948ae35de7e0f2b65e6d0653058f10))
* Add fixed-envelope rejection sampler ([6c7b5f0](https://github.com/denehoffman/laddu/commit/6c7b5f071337f7a7bc0e805a35c90abae470f254))
* Add generated batch metadata ([656fdfa](https://github.com/denehoffman/laddu/commit/656fdfa947d09a346c30346d143bf7e6c179ad91))
* Add generated layout query helpers ([490f8f3](https://github.com/denehoffman/laddu/commit/490f8f34e0c9ec54a79132f261d32c4872c129c5))
* Add generated p4 storage projection ([c9285b0](https://github.com/denehoffman/laddu/commit/c9285b04f0c8fc822f1317639d89d371b641762a))
* Add generated particle layout ids ([533ed3d](https://github.com/denehoffman/laddu/commit/533ed3dd43eb6dfb17f976ca33ed0d610c2114d0))
* Add generated particle species metadata ([e8bc310](https://github.com/denehoffman/laddu/commit/e8bc3109400af5c4ee7a4bab34b756d7d9ea4744))
* Add generation crate to workspace ([aa19e3f](https://github.com/denehoffman/laddu/commit/aa19e3ff9c5fb716a5c0767b57fb9a5ad3837869))
* Add metadata-aware fitting and closure projections ([f6d7857](https://github.com/denehoffman/laddu/commit/f6d78577b2c2f6cb6f6d707e58be03e8a9327eca))
* Add reusable histogram api ([30c11b0](https://github.com/denehoffman/laddu/commit/30c11b08990cf63b1c465be916925d51ddcc8559))
* Add vertex layouts to generation ([2cecde7](https://github.com/denehoffman/laddu/commit/2cecde7991273590e669db678e90e1e77d2f82ca))
* Clarify dataset event APIs ([ea99bad](https://github.com/denehoffman/laddu/commit/ea99bad056f5adda8a105aaac46b6d78ba84aa28))
* Derive generation from channel annotations ([ca84552](https://github.com/denehoffman/laddu/commit/ca845525e0eb875aa99a4a08d392fa6019866862))
* Expose direct ganesh fit and generation APIs ([4525a01](https://github.com/denehoffman/laddu/commit/4525a01feac3855232de9a2aad7713336d90dd3c))
* Expose generation bindings to python ([c2d1a5a](https://github.com/denehoffman/laddu/commit/c2d1a5a7b2ccba6ae962c2169c9c18ae2483f5b7))
* **fit:** Add generation closure workflow ([4556eaa](https://github.com/denehoffman/laddu/commit/4556eaa4c358ad05beee3482e218354108a75f63))
* **generation:** Add channel-driven event generation ([bdc0bd2](https://github.com/denehoffman/laddu/commit/bdc0bd212cbc72cee6637d9ddf78f1db59549737))
* **generation:** Add envelope policies ([e635d39](https://github.com/denehoffman/laddu/commit/e635d39eac6dd1c6f3c79d9ce13e7c93a1708d6a))
* **generation:** Add file sinks ([375751c](https://github.com/denehoffman/laddu/commit/375751cccb79bd158beefba9c0d78241efa955e6))
* **generation:** Add fixed-envelope rejection mode ([c79f25e](https://github.com/denehoffman/laddu/commit/c79f25e11f450efce0f156db1e58c0519d9f7a8e))
* **generation:** Add weighted generation mode ([f04cfd8](https://github.com/denehoffman/laddu/commit/f04cfd8b96318624845e452175db7e247e098bea))
* **generation:** Replace dataset generation with sinks ([b150c66](https://github.com/denehoffman/laddu/commit/b150c66da3c914ab2967eaec40e62eeadb6a8f20))
* **generation:** Support native python sinks ([7e3c3e5](https://github.com/denehoffman/laddu/commit/7e3c3e51d3a6cf7412185c0e6cd1f43bb2595e91))
* Major rewrite of laddu to use expression trees for amplitudes ([5774f78](https://github.com/denehoffman/laddu/commit/5774f78b20b7a351222f8bc5e6bba1e154a7566a))
* Make dataset access explicit ([7c003d4](https://github.com/denehoffman/laddu/commit/7c003d460ad87da00dffef2e80d3de2e8bda5ad4))
* Make memory budgets first-class ([171e658](https://github.com/denehoffman/laddu/commit/171e658ed8cc1a69330f9a68d488e3061e498a55))
* **physics:** Unify four-vector component order ([ad50eca](https://github.com/denehoffman/laddu/commit/ad50eca17b6a3e992494deb6b2794a8acd6e97dc))
* **python:** Add native package ecosystem ([20df1fb](https://github.com/denehoffman/laddu/commit/20df1fbad711534e84564040bef5b659c2a034a1))
* Query reaction particles by id ([a7a2938](https://github.com/denehoffman/laddu/commit/a7a29385cfe84cef3198aa71c090b3f3c2c5a86e))
* Redesign channel topology and generation APIs ([9d3e3a4](https://github.com/denehoffman/laddu/commit/9d3e3a44467a6ac14192e064430b1e4932770714))
* Redesign generation particle API ([48a9857](https://github.com/denehoffman/laddu/commit/48a9857f6db59b530f2de2434aa7420b60dc4a57))
* Replace legacy laddu code with expression-based rewrite ([1ccc1a3](https://github.com/denehoffman/laddu/commit/1ccc1a3c000adcd7d8e41182bcc8f0b72401b0cc))
* Rework kinematics API to allow for full Helicity/Canonical coupling terms, and other changes and improvements ([d6d49a5](https://github.com/denehoffman/laddu/commit/d6d49a5a66da6119b286c73637951954d77c2efb))
* Validate generation topology ([4abfb2e](https://github.com/denehoffman/laddu/commit/4abfb2e4aa6244243f499854dc973e11e18be88b))


### Bug Fixes

* **bench:** Raise generation memory budget ([2128b0d](https://github.com/denehoffman/laddu/commit/2128b0d0077956dcabf4263614101ba606f2842a))
* Clear clippy lints, especially Errors/Panics sections in docstrings ([2f998db](https://github.com/denehoffman/laddu/commit/2f998db89faeeed9ed70822c3a1d25081873e1b1))
* Generating Mandelstam-t from an exponential distribution produced unphysical (positive) values due to sign error ([f221a71](https://github.com/denehoffman/laddu/commit/f221a7188c7ad6aca2cbdd49bcc779f3200ad2be))
* Use truncated t distributions in generation ([3961faf](https://github.com/denehoffman/laddu/commit/3961fafe40d56b75eb20be294760a3ce91b328e8))


### Performance Improvements

* **generation:** Accelerate adaptive event sampling ([c978429](https://github.com/denehoffman/laddu/commit/c9784296cb89c2d1618a4336f741ab2f0d59b141))


### Miscellaneous Chores

* Remove legacy tree after migration ([cf43319](https://github.com/denehoffman/laddu/commit/cf43319bbee4357e2697857bdb2be69edfd10a83))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * laddu-compile bumped from 0.20.0 to 0.21.0
    * laddu-data bumped from 0.20.0 to 0.21.0
    * laddu-expr bumped from 0.20.0 to 0.21.0
    * laddu-physics bumped from 0.20.0 to 0.21.0
    * laddu-runtime bumped from 0.20.0 to 0.21.0
</details>

<details><summary>laddu-kernel: 0.21.0</summary>

## [0.21.0](https://github.com/denehoffman/laddu/compare/laddu-kernel-v0.20.0...laddu-kernel-v0.21.0) (2026-07-29)


###   BREAKING CHANGES

* major rewrite of laddu to use expression trees for amplitudes
* replace legacy laddu code with expression-based rewrite
* **runtime:** add full primal CPU JIT
* **expr:** rebuild expression graph pipeline
* replace the old laddu implementation with the new typed-kernel rewrite foundation.
* the normalization of standard relativistic Breit-Wigner amplitude is now unity.

### Features

* **autodiff:** Add backend-neutral gradient IR ([e27f185](https://github.com/denehoffman/laddu/commit/e27f18554a8a3f15d301ffc00754ee2804c4cc85))
* **expr:** Rebuild expression graph pipeline ([fa3a3ed](https://github.com/denehoffman/laddu/commit/fa3a3ed71a1b76f118b41813489d17e0a2f82590))
* **kernel:** Add cache materialization ir ([7c8f774](https://github.com/denehoffman/laddu/commit/7c8f774e5ec5ecae6f7009a25069bdc94e255844))
* **kernel:** Add scalar execution ir ([73e4a68](https://github.com/denehoffman/laddu/commit/73e4a686343d3aca2d2c4fbab501f4f9e65138d8))
* Major rewrite of laddu to use expression trees for amplitudes ([5774f78](https://github.com/denehoffman/laddu/commit/5774f78b20b7a351222f8bc5e6bba1e154a7566a))
* **python:** Add native package ecosystem ([20df1fb](https://github.com/denehoffman/laddu/commit/20df1fbad711534e84564040bef5b659c2a034a1))
* Replace legacy laddu code with expression-based rewrite ([1ccc1a3](https://github.com/denehoffman/laddu/commit/1ccc1a3c000adcd7d8e41182bcc8f0b72401b0cc))
* Rework kinematics API to allow for full Helicity/Canonical coupling terms, and other changes and improvements ([d6d49a5](https://github.com/denehoffman/laddu/commit/d6d49a5a66da6119b286c73637951954d77c2efb))
* **runtime:** Add full primal CPU JIT ([58c40c9](https://github.com/denehoffman/laddu/commit/58c40c942b49ca4c4502165bdc5d8ace4c80fa08))
* **runtime:** Add scalar executor selection ([ef7b1cd](https://github.com/denehoffman/laddu/commit/ef7b1cd05a2a6550a1202670849fab4ecc566e96))
* **runtime:** Unify gradient kernel execution ([16a4fc0](https://github.com/denehoffman/laddu/commit/16a4fc08598839198713d9e9002d6ee8e583e9a0))
* Start breaking laddu kernel rewrite ([8051e64](https://github.com/denehoffman/laddu/commit/8051e6444bba62877abdb62c2a06fbd002a0334d))


### Bug Fixes

* Clear clippy lints, especially Errors/Panics sections in docstrings ([2f998db](https://github.com/denehoffman/laddu/commit/2f998db89faeeed9ed70822c3a1d25081873e1b1))
* Reorganize all crates and provide convenience methods for dealing with parameters ([1751265](https://github.com/denehoffman/laddu/commit/1751265ba5473cad0a92eedbde6286afe60c8b92))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * laddu-expr bumped from 0.20.0 to 0.21.0
</details>

<details><summary>laddu-likelihood: 0.21.0</summary>

## [0.21.0](https://github.com/denehoffman/laddu/compare/laddu-likelihood-v0.20.0...laddu-likelihood-v0.21.0) (2026-07-29)


###   BREAKING CHANGES

* major rewrite of laddu to use expression trees for amplitudes
* replace legacy laddu code with expression-based rewrite
* make memory budgets first-class
* expose direct ganesh fit and generation APIs
* **runtime:** integrate WGPU likelihood execution
* **runtime:** unify execution and likelihood APIs
* **runtime:** add compiled reduction plans
* **runtime:** unify dataset execution policies
* **likelihood:** port K-matrix NLL benchmark
* replace the old laddu implementation with the new typed-kernel rewrite foundation.
* the normalization of standard relativistic Breit-Wigner amplitude is now unity.

### Features

* Add bootstrapped cross-section workflow ([c924c4e](https://github.com/denehoffman/laddu/commit/c924c4e71bea157af4d1bd34e8f6fec41b40f7a1))
* Add extended NLL and parameter recompilation ([4055c16](https://github.com/denehoffman/laddu/commit/4055c16d6cfa6ba6cee740174ceb589d65aa976f))
* Add lazy dataset queries and projections ([fbfde99](https://github.com/denehoffman/laddu/commit/fbfde995b7903e04833b1633cffb9f458d8f6f1d))
* **autodiff:** Add forward gradients ([c98dba9](https://github.com/denehoffman/laddu/commit/c98dba948578b4c1d95d98598e5717d7cea671a2))
* Expose direct ganesh fit and generation APIs ([4525a01](https://github.com/denehoffman/laddu/commit/4525a01feac3855232de9a2aad7713336d90dd3c))
* **fit:** Integrate ganesh optimization and sampling ([79abcca](https://github.com/denehoffman/laddu/commit/79abcca5e12aa734c2616e93bcce6fc52cf3eed2))
* Improve public Rust API ergonomics ([86ce01b](https://github.com/denehoffman/laddu/commit/86ce01b6758e4db875aec021e40d9ed635f02199))
* **likelihood:** Add cached normalized intensity fits ([6b90132](https://github.com/denehoffman/laddu/commit/6b90132584c4a3b7964164b20b8884924c948150))
* **likelihood:** Add cross-section analysis API ([dd34c80](https://github.com/denehoffman/laddu/commit/dd34c80fd7f2e049369d285eeaa2b8c5aa66886d))
* **likelihood:** Support custom additive terms ([1d33285](https://github.com/denehoffman/laddu/commit/1d332858adb80ffe69403a98346db7f4a33c9dd4))
* Major rewrite of laddu to use expression trees for amplitudes ([5774f78](https://github.com/denehoffman/laddu/commit/5774f78b20b7a351222f8bc5e6bba1e154a7566a))
* Make memory budgets first-class ([171e658](https://github.com/denehoffman/laddu/commit/171e658ed8cc1a69330f9a68d488e3061e498a55))
* Periodic parameters, objectives, and a between query, as well as organizational changes to prepare for Python API ([b1e004f](https://github.com/denehoffman/laddu/commit/b1e004f1aa0b8c16075bbb1e19c580377ecc3317))
* **python:** Add native package ecosystem ([20df1fb](https://github.com/denehoffman/laddu/commit/20df1fbad711534e84564040bef5b659c2a034a1))
* Replace legacy laddu code with expression-based rewrite ([1ccc1a3](https://github.com/denehoffman/laddu/commit/1ccc1a3c000adcd7d8e41182bcc8f0b72401b0cc))
* Rework kinematics API to allow for full Helicity/Canonical coupling terms, and other changes and improvements ([d6d49a5](https://github.com/denehoffman/laddu/commit/d6d49a5a66da6119b286c73637951954d77c2efb))
* **runtime:** Add compiled reduction plans ([aaff110](https://github.com/denehoffman/laddu/commit/aaff1103a7986f45dff0464a899487e57fb3ecb7))
* **runtime:** Add initial f32 scalar execution ([585d979](https://github.com/denehoffman/laddu/commit/585d9796ae112d74abbae051226a2d4b9726ef6f))
* **runtime:** Allow distributed wgpu execution ([e4a5136](https://github.com/denehoffman/laddu/commit/e4a5136280b0600d02a4e2df5fde70b12cc07fe1))
* **runtime:** Complete CPU gradient JIT parity ([2dc038a](https://github.com/denehoffman/laddu/commit/2dc038abe4eba9d67f3b5cc75f680075ca8166c4))
* **runtime:** Integrate WGPU likelihood execution ([05673fc](https://github.com/denehoffman/laddu/commit/05673fcee7dc5e44c7bac952e0681d630ca94f11))
* **runtime:** Unify dataset execution policies ([3c85961](https://github.com/denehoffman/laddu/commit/3c85961909dab88f82cb2ff2d37171dc6e2c5408))
* Start breaking laddu kernel rewrite ([8051e64](https://github.com/denehoffman/laddu/commit/8051e6444bba62877abdb62c2a06fbd002a0334d))
* **wgpu:** Add scalar gradient reductions ([9061b6a](https://github.com/denehoffman/laddu/commit/9061b6a3920253add7ee36c37d908501ebeb6253))
* **wgpu:** Support aggregate algebra ([8c540f4](https://github.com/denehoffman/laddu/commit/8c540f469bc9b87ffb564ca3c99c4704e995b47f))


### Bug Fixes

* Change ExprId to u64 to remove some expects and clear all clippy lints ([985c8b8](https://github.com/denehoffman/laddu/commit/985c8b840b56a7df644ca4c8da151921664d62de))
* Change where errors are thrown for unsupported f32 execution ([e9e9227](https://github.com/denehoffman/laddu/commit/e9e9227e6ed74929ddec9c24f28c9e0dd727556b))
* Clear clippy lints, especially Errors/Panics sections in docstrings ([2f998db](https://github.com/denehoffman/laddu/commit/2f998db89faeeed9ed70822c3a1d25081873e1b1))
* Reorganize all crates and provide convenience methods for dealing with parameters ([1751265](https://github.com/denehoffman/laddu/commit/1751265ba5473cad0a92eedbde6286afe60c8b92))


### Performance Improvements

* **likelihood:** Port K-matrix NLL benchmark ([05d213d](https://github.com/denehoffman/laddu/commit/05d213d8d10ac73b8311af7a6c13e01395f2334d))


### Code Refactoring

* **runtime:** Unify execution and likelihood APIs ([758b038](https://github.com/denehoffman/laddu/commit/758b03884caa0d690fb9beaa9f2e270c72c24a81))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * laddu-compile bumped from 0.20.0 to 0.21.0
    * laddu-data bumped from 0.20.0 to 0.21.0
    * laddu-expr bumped from 0.20.0 to 0.21.0
    * laddu-runtime bumped from 0.20.0 to 0.21.0
  * dev-dependencies
    * laddu-amplitudes bumped from 0.20.0 to 0.21.0
</details>

<details><summary>laddu-memory: 0.21.0</summary>

## [0.21.0](https://github.com/denehoffman/laddu/compare/laddu-memory-v0.20.0...laddu-memory-v0.21.0) (2026-07-29)


###   BREAKING CHANGES

* major rewrite of laddu to use expression trees for amplitudes
* replace legacy laddu code with expression-based rewrite
* make memory budgets first-class
* the normalization of standard relativistic Breit-Wigner amplitude is now unity.

### Features

* Major rewrite of laddu to use expression trees for amplitudes ([5774f78](https://github.com/denehoffman/laddu/commit/5774f78b20b7a351222f8bc5e6bba1e154a7566a))
* Make memory budgets first-class ([171e658](https://github.com/denehoffman/laddu/commit/171e658ed8cc1a69330f9a68d488e3061e498a55))
* Replace legacy laddu code with expression-based rewrite ([1ccc1a3](https://github.com/denehoffman/laddu/commit/1ccc1a3c000adcd7d8e41182bcc8f0b72401b0cc))
* Rework kinematics API to allow for full Helicity/Canonical coupling terms, and other changes and improvements ([d6d49a5](https://github.com/denehoffman/laddu/commit/d6d49a5a66da6119b286c73637951954d77c2efb))
</details>

<details><summary>laddu-physics: 0.21.0</summary>

## [0.21.0](https://github.com/denehoffman/laddu/compare/laddu-physics-v0.20.0...laddu-physics-v0.21.0) (2026-07-29)


###   BREAKING CHANGES

* major rewrite of laddu to use expression trees for amplitudes
* replace legacy laddu code with expression-based rewrite
* **physics:** Four-vector constructors, conversions, public fields, and positional arrays now use (E, px, py, pz) order.
* **physics:** align histogram construction with NumPy
* expose direct ganesh fit and generation APIs
* add metadata-aware fitting and closure projections
* **generation:** UnweightedConfig::new now takes only the requested event count, and max_proposals is Option<usize>; use with_max_proposals to opt into a limit.
* **generation:** remove FixedInitialState and InitialStateSampler; initial momentum sources now belong to channel edges and ChannelGenerator::new accepts only a Channel.
* **amplitudes:** rename scalar kinematics and Breit-Wigner helpers to the expression-oriented API.
* move Breit-Wigner functions to amplitudes
* **physics:** add expression vector builders
* replace the old laddu implementation with the new typed-kernel rewrite foundation.
* the normalization of standard relativistic Breit-Wigner amplitude is now unity.

### Features

* Add channel kinematics and p4 event expressions ([d9d4c81](https://github.com/denehoffman/laddu/commit/d9d4c813cdc434c3097ed085a679e82afa3234f9))
* Add histogram uncertainty controls ([17ff935](https://github.com/denehoffman/laddu/commit/17ff9351508dd92b158d2dc643b4342d3b43794f))
* Add metadata-aware fitting and closure projections ([f6d7857](https://github.com/denehoffman/laddu/commit/f6d78577b2c2f6cb6f6d707e58be03e8a9327eca))
* **amplitudes:** Add composable K-matrix amplitudes ([07edf50](https://github.com/denehoffman/laddu/commit/07edf5034a82b9a7d9521ad1cdf54a6ea5af8199))
* Expose direct ganesh fit and generation APIs ([4525a01](https://github.com/denehoffman/laddu/commit/4525a01feac3855232de9a2aad7713336d90dd3c))
* **generation:** Add channel-driven event generation ([bdc0bd2](https://github.com/denehoffman/laddu/commit/bdc0bd212cbc72cee6637d9ddf78f1db59549737))
* Improve public Rust API ergonomics ([86ce01b](https://github.com/denehoffman/laddu/commit/86ce01b6758e4db875aec021e40d9ed635f02199))
* Major rewrite of laddu to use expression trees for amplitudes ([5774f78](https://github.com/denehoffman/laddu/commit/5774f78b20b7a351222f8bc5e6bba1e154a7566a))
* **physics:** Add expression vector builders ([09c0552](https://github.com/denehoffman/laddu/commit/09c0552be110480d9705d92ce23797a50710c143))
* **physics:** Align histogram construction with NumPy ([58e1565](https://github.com/denehoffman/laddu/commit/58e156534d10d3cd0e930319fffea6d1234bfa62))
* **physics:** Unify four-vector component order ([ad50eca](https://github.com/denehoffman/laddu/commit/ad50eca17b6a3e992494deb6b2794a8acd6e97dc))
* **python:** Add native package ecosystem ([20df1fb](https://github.com/denehoffman/laddu/commit/20df1fbad711534e84564040bef5b659c2a034a1))
* **quantum:** Expose selection rules in Python ([9091f5b](https://github.com/denehoffman/laddu/commit/9091f5b551ef87b300835cb800051fbc9ef5efc8))
* Replace legacy laddu code with expression-based rewrite ([1ccc1a3](https://github.com/denehoffman/laddu/commit/1ccc1a3c000adcd7d8e41182bcc8f0b72401b0cc))
* Rework kinematics API to allow for full Helicity/Canonical coupling terms, and other changes and improvements ([d6d49a5](https://github.com/denehoffman/laddu/commit/d6d49a5a66da6119b286c73637951954d77c2efb))
* Start breaking laddu kernel rewrite ([8051e64](https://github.com/denehoffman/laddu/commit/8051e6444bba62877abdb62c2a06fbd002a0334d))


### Bug Fixes

* Change ExprId to u64 to remove some expects and clear all clippy lints ([985c8b8](https://github.com/denehoffman/laddu/commit/985c8b840b56a7df644ca4c8da151921664d62de))
* **ci:** Pass pre-push verification ([367c169](https://github.com/denehoffman/laddu/commit/367c16951e8792c81f1203ce6af7ba321b5c0e5d))
* Clear clippy lints, especially Errors/Panics sections in docstrings ([2f998db](https://github.com/denehoffman/laddu/commit/2f998db89faeeed9ed70822c3a1d25081873e1b1))
* **release:** Break dev dependency cycles ([c6f97c0](https://github.com/denehoffman/laddu/commit/c6f97c0ba5ed60daa56b6a5ea8dc71142fd8c0da))
* Reorganize all crates and provide convenience methods for dealing with parameters ([1751265](https://github.com/denehoffman/laddu/commit/1751265ba5473cad0a92eedbde6286afe60c8b92))


### Performance Improvements

* **generation:** Accelerate adaptive event sampling ([c978429](https://github.com/denehoffman/laddu/commit/c9784296cb89c2d1618a4336f741ab2f0d59b141))


### Code Refactoring

* Move Breit-Wigner functions to amplitudes ([e990266](https://github.com/denehoffman/laddu/commit/e990266160ff779928411a73358b6c906b4a6a02))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * laddu-expr bumped from 0.20.0 to 0.21.0
  * dev-dependencies
    * laddu-compile bumped from 0.20.0 to 0.21.0
</details>

<details><summary>laddu-python: 0.21.0</summary>

## [0.21.0](https://github.com/denehoffman/laddu/compare/laddu-python-v0.20.0...laddu-python-v0.21.0) (2026-07-29)


###   BREAKING CHANGES

* major rewrite of laddu to use expression trees for amplitudes
* replace legacy laddu code with expression-based rewrite
* the normalization of standard relativistic Breit-Wigner amplitude is now unity.

### Features

* Major rewrite of laddu to use expression trees for amplitudes ([5774f78](https://github.com/denehoffman/laddu/commit/5774f78b20b7a351222f8bc5e6bba1e154a7566a))
* **python:** Add native package ecosystem ([20df1fb](https://github.com/denehoffman/laddu/commit/20df1fbad711534e84564040bef5b659c2a034a1))
* Replace legacy laddu code with expression-based rewrite ([1ccc1a3](https://github.com/denehoffman/laddu/commit/1ccc1a3c000adcd7d8e41182bcc8f0b72401b0cc))
* Rework kinematics API to allow for full Helicity/Canonical coupling terms, and other changes and improvements ([d6d49a5](https://github.com/denehoffman/laddu/commit/d6d49a5a66da6119b286c73637951954d77c2efb))


### Bug Fixes

* **release:** Synchronize aliased Python dependency ([d2ea1e1](https://github.com/denehoffman/laddu/commit/d2ea1e1ee4c1cb1970f966d7fa801dd1ba459b6f))
</details>

<details><summary>laddu-python-local: 0.21.0</summary>

## [0.21.0](https://github.com/denehoffman/laddu/compare/laddu-python-local-v0.20.0...laddu-python-local-v0.21.0) (2026-07-29)


###   BREAKING CHANGES

* major rewrite of laddu to use expression trees for amplitudes
* replace legacy laddu code with expression-based rewrite
* the normalization of standard relativistic Breit-Wigner amplitude is now unity.

### Features

* Major rewrite of laddu to use expression trees for amplitudes ([5774f78](https://github.com/denehoffman/laddu/commit/5774f78b20b7a351222f8bc5e6bba1e154a7566a))
* **python:** Add native package ecosystem ([20df1fb](https://github.com/denehoffman/laddu/commit/20df1fbad711534e84564040bef5b659c2a034a1))
* Replace legacy laddu code with expression-based rewrite ([1ccc1a3](https://github.com/denehoffman/laddu/commit/1ccc1a3c000adcd7d8e41182bcc8f0b72401b0cc))
* Rework kinematics API to allow for full Helicity/Canonical coupling terms, and other changes and improvements ([d6d49a5](https://github.com/denehoffman/laddu/commit/d6d49a5a66da6119b286c73637951954d77c2efb))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * laddu bumped from 0.20.0 to 0.20.1
</details>

<details><summary>laddu-python-mpi: 0.21.0</summary>

## [0.21.0](https://github.com/denehoffman/laddu/compare/laddu-python-mpi-v0.20.0...laddu-python-mpi-v0.21.0) (2026-07-29)


###   BREAKING CHANGES

* major rewrite of laddu to use expression trees for amplitudes
* replace legacy laddu code with expression-based rewrite
* the normalization of standard relativistic Breit-Wigner amplitude is now unity.

### Features

* Major rewrite of laddu to use expression trees for amplitudes ([5774f78](https://github.com/denehoffman/laddu/commit/5774f78b20b7a351222f8bc5e6bba1e154a7566a))
* **python:** Add native package ecosystem ([20df1fb](https://github.com/denehoffman/laddu/commit/20df1fbad711534e84564040bef5b659c2a034a1))
* Replace legacy laddu code with expression-based rewrite ([1ccc1a3](https://github.com/denehoffman/laddu/commit/1ccc1a3c000adcd7d8e41182bcc8f0b72401b0cc))
* Rework kinematics API to allow for full Helicity/Canonical coupling terms, and other changes and improvements ([d6d49a5](https://github.com/denehoffman/laddu/commit/d6d49a5a66da6119b286c73637951954d77c2efb))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * laddu bumped from 0.20.0 to 0.20.1
</details>

<details><summary>laddu-runtime: 0.21.0</summary>

## [0.21.0](https://github.com/denehoffman/laddu/compare/laddu-runtime-v0.20.0...laddu-runtime-v0.21.0) (2026-07-29)


###   BREAKING CHANGES

* major rewrite of laddu to use expression trees for amplitudes
* replace legacy laddu code with expression-based rewrite
* make memory budgets first-class
* **physics:** Four-vector constructors, conversions, public fields, and positional arrays now use (E, px, py, pz) order.
* **runtime:** integrate WGPU likelihood execution
* **runtime:** unify execution and likelihood APIs
* **runtime:** add compiled reduction plans
* **runtime:** unify dataset execution policies
* **runtime:** add full primal CPU JIT
* **expr:** lower complex parameters to expressions
* **compile:** normalize n-ary algebra
* **compile:** the default optimizer now reassociates canonical Add/Mul trees and merges exp products, further changing optimized graph shape and floating-point operation order.
* **compile:** expression graphs now include first-class Complex nodes and the default compile pipeline performs aggressive canonicalization/CSE.
* replace the old laddu implementation with the new typed-kernel rewrite foundation.
* the normalization of standard relativistic Breit-Wigner amplitude is now unity.

### Features

* Add channel kinematics and p4 event expressions ([d9d4c81](https://github.com/denehoffman/laddu/commit/d9d4c813cdc434c3097ed085a679e82afa3234f9))
* Add lazy dataset queries and projections ([fbfde99](https://github.com/denehoffman/laddu/commit/fbfde995b7903e04833b1633cffb9f458d8f6f1d))
* Add serde support to public API types ([9e0ec82](https://github.com/denehoffman/laddu/commit/9e0ec82936c27ec070587ddf8b4e3ae8d16d7acd))
* **autodiff:** Add forward gradients ([c98dba9](https://github.com/denehoffman/laddu/commit/c98dba948578b4c1d95d98598e5717d7cea671a2))
* **compile:** Add canonical CSE and complex IR ([8c660ca](https://github.com/denehoffman/laddu/commit/8c660ca657c3e52e97e31bcbc6f4caf45740d20c))
* **compile:** Merge exponential products ([5e91de5](https://github.com/denehoffman/laddu/commit/5e91de5ad9d2a0df6a293d7e6e14bed9bafcf144))
* **compile:** Normalize n-ary algebra ([8bdbd74](https://github.com/denehoffman/laddu/commit/8bdbd74718b72dafd7685ec25ef4b2ec2b932346))
* **kernel:** Add scalar execution ir ([73e4a68](https://github.com/denehoffman/laddu/commit/73e4a686343d3aca2d2c4fbab501f4f9e65138d8))
* **likelihood:** Add cached normalized intensity fits ([6b90132](https://github.com/denehoffman/laddu/commit/6b90132584c4a3b7964164b20b8884924c948150))
* Major rewrite of laddu to use expression trees for amplitudes ([5774f78](https://github.com/denehoffman/laddu/commit/5774f78b20b7a351222f8bc5e6bba1e154a7566a))
* Make memory budgets first-class ([171e658](https://github.com/denehoffman/laddu/commit/171e658ed8cc1a69330f9a68d488e3061e498a55))
* Periodic parameters, objectives, and a between query, as well as organizational changes to prepare for Python API ([b1e004f](https://github.com/denehoffman/laddu/commit/b1e004f1aa0b8c16075bbb1e19c580377ecc3317))
* **physics:** Unify four-vector component order ([ad50eca](https://github.com/denehoffman/laddu/commit/ad50eca17b6a3e992494deb6b2794a8acd6e97dc))
* **python:** Add native package ecosystem ([20df1fb](https://github.com/denehoffman/laddu/commit/20df1fbad711534e84564040bef5b659c2a034a1))
* Replace legacy laddu code with expression-based rewrite ([1ccc1a3](https://github.com/denehoffman/laddu/commit/1ccc1a3c000adcd7d8e41182bcc8f0b72401b0cc))
* Rework kinematics API to allow for full Helicity/Canonical coupling terms, and other changes and improvements ([d6d49a5](https://github.com/denehoffman/laddu/commit/d6d49a5a66da6119b286c73637951954d77c2efb))
* **runtime:** Add compiled reduction plans ([aaff110](https://github.com/denehoffman/laddu/commit/aaff1103a7986f45dff0464a899487e57fb3ecb7))
* **runtime:** Add CPU graph evaluator ([c36b6bc](https://github.com/denehoffman/laddu/commit/c36b6bcd459c23f95c1061877830cda5f730dbc2))
* **runtime:** Add dataset-resident event caches ([fa52ebc](https://github.com/denehoffman/laddu/commit/fa52ebcdf2c52c544a1f79b2e7fc96a73b1793a3))
* **runtime:** Add f32 cpu scalar gradients ([d118caa](https://github.com/denehoffman/laddu/commit/d118caa35b8fd91f0b84fd0bbd2876329a96371e))
* **runtime:** Add f32 scalar jit ([d567fe6](https://github.com/denehoffman/laddu/commit/d567fe6c399cf216c2f9e44fc00ff908f508df86))
* **runtime:** Add f64 cpu reverse autograd ([75cd3c9](https://github.com/denehoffman/laddu/commit/75cd3c9838f809e37412b18a368cf6578152c545))
* **runtime:** Add full primal CPU JIT ([58c40c9](https://github.com/denehoffman/laddu/commit/58c40c942b49ca4c4502165bdc5d8ace4c80fa08))
* **runtime:** Add initial f32 scalar execution ([585d979](https://github.com/denehoffman/laddu/commit/585d9796ae112d74abbae051226a2d4b9726ef6f))
* **runtime:** Add JIT gradient execution ([4ec096a](https://github.com/denehoffman/laddu/commit/4ec096a757750a9879fda1c99def667bcbda6db5))
* **runtime:** Add scalar executor selection ([ef7b1cd](https://github.com/denehoffman/laddu/commit/ef7b1cd05a2a6550a1202670849fab4ecc566e96))
* **runtime:** Add typed cache layouts ([d4108ba](https://github.com/denehoffman/laddu/commit/d4108ba3a6646b9d94e21ac852680dfa25a88a50))
* **runtime:** Allow distributed wgpu execution ([e4a5136](https://github.com/denehoffman/laddu/commit/e4a5136280b0600d02a4e2df5fde70b12cc07fe1))
* **runtime:** Complete CPU gradient JIT parity ([2dc038a](https://github.com/denehoffman/laddu/commit/2dc038abe4eba9d67f3b5cc75f680075ca8166c4))
* **runtime:** Integrate WGPU likelihood execution ([05673fc](https://github.com/denehoffman/laddu/commit/05673fcee7dc5e44c7bac952e0681d630ca94f11))
* **runtime:** Support f32 direct event gradients ([65230dd](https://github.com/denehoffman/laddu/commit/65230dd03bc6791c0ebcffd5839328d2c207a241))
* **runtime:** Support JIT gradients through solves ([e2d2ca3](https://github.com/denehoffman/laddu/commit/e2d2ca35c7e357737a49692d8698a760964002af))
* **runtime:** Unify dataset execution policies ([3c85961](https://github.com/denehoffman/laddu/commit/3c85961909dab88f82cb2ff2d37171dc6e2c5408))
* **runtime:** Unify gradient kernel execution ([16a4fc0](https://github.com/denehoffman/laddu/commit/16a4fc08598839198713d9e9002d6ee8e583e9a0))
* Start breaking laddu kernel rewrite ([8051e64](https://github.com/denehoffman/laddu/commit/8051e6444bba62877abdb62c2a06fbd002a0334d))
* **wgpu:** Add scalar gradient reductions ([9061b6a](https://github.com/denehoffman/laddu/commit/9061b6a3920253add7ee36c37d908501ebeb6253))


### Bug Fixes

* Change ExprId to u64 to remove some expects and clear all clippy lints ([985c8b8](https://github.com/denehoffman/laddu/commit/985c8b840b56a7df644ca4c8da151921664d62de))
* Change where errors are thrown for unsupported f32 execution ([e9e9227](https://github.com/denehoffman/laddu/commit/e9e9227e6ed74929ddec9c24f28c9e0dd727556b))
* Clean up benchmarks, tests, and a few other gradient-related spots ([612376b](https://github.com/denehoffman/laddu/commit/612376bbbde51b86e28c143e14a55dceaebc716e))
* Clear clippy lints, especially Errors/Panics sections in docstrings ([2f998db](https://github.com/denehoffman/laddu/commit/2f998db89faeeed9ed70822c3a1d25081873e1b1))
* Reorganize all crates and provide convenience methods for dealing with parameters ([1751265](https://github.com/denehoffman/laddu/commit/1751265ba5473cad0a92eedbde6286afe60c8b92))


### Performance Improvements

* **compile:** Cache only event frontier nodes ([546e9de](https://github.com/denehoffman/laddu/commit/546e9de4f3df6d8f72bba6564b94e6fcb3e2274e))
* **runtime:** Avoid allocations in small JIT solves ([f3cc701](https://github.com/denehoffman/laddu/commit/f3cc701d5d5d38082644e6207f7933ecc06393ca))
* **runtime:** Cache selected solve rows ([a613067](https://github.com/denehoffman/laddu/commit/a613067cfa345442918069f5cdd41d9acbd86b09))
* **runtime:** Compact cached evaluation storage ([e49ad6a](https://github.com/denehoffman/laddu/commit/e49ad6a81222c1c39328c037cb12daa303bf62fc))
* **runtime:** Evaluate scalar events in blocks ([c463a2c](https://github.com/denehoffman/laddu/commit/c463a2c3b839a7334ace985059f6513cd1aa3e68))
* **runtime:** Execute scalar graphs with typed tape ([b5eac31](https://github.com/denehoffman/laddu/commit/b5eac31bb0e70a16954de6e9bd0831e1a142ac07))
* **runtime:** Hoist invariant scalar instructions ([643ba13](https://github.com/denehoffman/laddu/commit/643ba132cd9ffe1ab2513407a946f39d35a85bc8))
* **runtime:** Hoist scalar operation dispatch ([4f4a45c](https://github.com/denehoffman/laddu/commit/4f4a45cca8686c566fa98706819522ee9a7f3451))
* **runtime:** Optimize CPU gradient JIT ([35e5b99](https://github.com/denehoffman/laddu/commit/35e5b992842ada13922c3b0418bbeeb547cebeb2))
* **runtime:** Reuse f32 gradient kernels ([d03118c](https://github.com/denehoffman/laddu/commit/d03118cd3ba5302ac991eaac88ec9233cfe9d507))
* **runtime:** Reuse scalar block slots ([bcc79c2](https://github.com/denehoffman/laddu/commit/bcc79c256579ce024f4e4581fafb4d67b9c0c45e))
* **runtime:** Reuse scalar event workspaces ([9b6ef3f](https://github.com/denehoffman/laddu/commit/9b6ef3f99dc3748d2296889e2d680229a7918e26))
* **runtime:** Scalarize selected solve inputs ([2f4a6d6](https://github.com/denehoffman/laddu/commit/2f4a6d6b030aa4874d1a6db32c4dcc1f2e96ecfb))
* **runtime:** Skip cached subgraphs during evaluation ([f6bba4b](https://github.com/denehoffman/laddu/commit/f6bba4b8d6b16a2277ff9f5c272ea40d8e89bdb8))
* **runtime:** Specialize real scalar tape ([e2a774b](https://github.com/denehoffman/laddu/commit/e2a774b3220f435fba00d75f5f97da62c4b1a0d4))
* **runtime:** Specialize scalar operand access ([d9d7d28](https://github.com/denehoffman/laddu/commit/d9d7d28dfe90e5535eba937c990ea840acf428ca))
* **runtime:** Use fixed scalar blocks ([af1d12c](https://github.com/denehoffman/laddu/commit/af1d12c180c78810e8334b218f70c90bacfa17b6))
* **runtime:** Validate caches once per batch ([11316f4](https://github.com/denehoffman/laddu/commit/11316f42d112db4fb350a96a9a9084d2b60b68df))


### Code Refactoring

* **expr:** Lower complex parameters to expressions ([5c528d2](https://github.com/denehoffman/laddu/commit/5c528d28aeb1afeb6678e76a8c329ad9f8efa453))
* **runtime:** Unify execution and likelihood APIs ([758b038](https://github.com/denehoffman/laddu/commit/758b03884caa0d690fb9beaa9f2e270c72c24a81))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * laddu-autodiff bumped from 0.20.0 to 0.21.0
    * laddu-compile bumped from 0.20.0 to 0.21.0
    * laddu-data bumped from 0.20.0 to 0.21.0
    * laddu-expr bumped from 0.20.0 to 0.21.0
    * laddu-kernel bumped from 0.20.0 to 0.21.0
    * laddu-memory bumped from 0.20.0 to 0.21.0
    * laddu-wgpu bumped from 0.20.0 to 0.21.0
</details>

<details><summary>laddu-wgpu: 0.21.0</summary>

## [0.21.0](https://github.com/denehoffman/laddu/compare/laddu-wgpu-v0.20.0...laddu-wgpu-v0.21.0) (2026-07-29)


###   BREAKING CHANGES

* major rewrite of laddu to use expression trees for amplitudes
* replace legacy laddu code with expression-based rewrite
* make memory budgets first-class
* **physics:** Four-vector constructors, conversions, public fields, and positional arrays now use (E, px, py, pz) order.
* expose direct ganesh fit and generation APIs
* **runtime:** integrate WGPU likelihood execution
* replace the old laddu implementation with the new typed-kernel rewrite foundation.
* the normalization of standard relativistic Breit-Wigner amplitude is now unity.

### Features

* Add f64 GPU pipeline with manual expansions for unsupported functions ([9f7e2db](https://github.com/denehoffman/laddu/commit/9f7e2db771af85918d3e8d53a182b6342afd210a))
* Add serde support to public API types ([9e0ec82](https://github.com/denehoffman/laddu/commit/9e0ec82936c27ec070587ddf8b4e3ae8d16d7acd))
* Expose direct ganesh fit and generation APIs ([4525a01](https://github.com/denehoffman/laddu/commit/4525a01feac3855232de9a2aad7713336d90dd3c))
* Improve public Rust API ergonomics ([86ce01b](https://github.com/denehoffman/laddu/commit/86ce01b6758e4db875aec021e40d9ed635f02199))
* **kernel:** Add cache materialization ir ([7c8f774](https://github.com/denehoffman/laddu/commit/7c8f774e5ec5ecae6f7009a25069bdc94e255844))
* Major rewrite of laddu to use expression trees for amplitudes ([5774f78](https://github.com/denehoffman/laddu/commit/5774f78b20b7a351222f8bc5e6bba1e154a7566a))
* Make memory budgets first-class ([171e658](https://github.com/denehoffman/laddu/commit/171e658ed8cc1a69330f9a68d488e3061e498a55))
* **physics:** Unify four-vector component order ([ad50eca](https://github.com/denehoffman/laddu/commit/ad50eca17b6a3e992494deb6b2794a8acd6e97dc))
* **python:** Add native package ecosystem ([20df1fb](https://github.com/denehoffman/laddu/commit/20df1fbad711534e84564040bef5b659c2a034a1))
* Replace legacy laddu code with expression-based rewrite ([1ccc1a3](https://github.com/denehoffman/laddu/commit/1ccc1a3c000adcd7d8e41182bcc8f0b72401b0cc))
* Rework kinematics API to allow for full Helicity/Canonical coupling terms, and other changes and improvements ([d6d49a5](https://github.com/denehoffman/laddu/commit/d6d49a5a66da6119b286c73637951954d77c2efb))
* **runtime:** Integrate WGPU likelihood execution ([05673fc](https://github.com/denehoffman/laddu/commit/05673fcee7dc5e44c7bac952e0681d630ca94f11))
* Start breaking laddu kernel rewrite ([8051e64](https://github.com/denehoffman/laddu/commit/8051e6444bba62877abdb62c2a06fbd002a0334d))
* **wgpu:** Add adapter discovery and device contexts ([c73a3b7](https://github.com/denehoffman/laddu/commit/c73a3b739396e98032f4b3cb1f84d93d1359aa38))
* **wgpu:** Add scalar gradient reductions ([9061b6a](https://github.com/denehoffman/laddu/commit/9061b6a3920253add7ee36c37d908501ebeb6253))
* **wgpu:** Evaluate and reduce event batches ([49c8c52](https://github.com/denehoffman/laddu/commit/49c8c52ad103a53a59199436bc884ee4e33ee92a))
* **wgpu:** Execute scalar kernels on gpu ([e107200](https://github.com/denehoffman/laddu/commit/e1072008ca4b08330ddee55d7d47399050a6b02c))
* **wgpu:** Materialize computed event caches ([7eb12aa](https://github.com/denehoffman/laddu/commit/7eb12aa147714c851b21029b9d7be8469afe492e))
* **wgpu:** Stream batches within memory budgets ([419baad](https://github.com/denehoffman/laddu/commit/419baad1bf76dc22f14d12e09de782256c3582b4))
* **wgpu:** Support aggregate algebra ([8c540f4](https://github.com/denehoffman/laddu/commit/8c540f469bc9b87ffb564ca3c99c4704e995b47f))


### Bug Fixes

* **ci:** Pass pre-push verification ([367c169](https://github.com/denehoffman/laddu/commit/367c16951e8792c81f1203ce6af7ba321b5c0e5d))
* Clear clippy lints, especially Errors/Panics sections in docstrings ([2f998db](https://github.com/denehoffman/laddu/commit/2f998db89faeeed9ed70822c3a1d25081873e1b1))
* **release:** Break dev dependency cycles ([c6f97c0](https://github.com/denehoffman/laddu/commit/c6f97c0ba5ed60daa56b6a5ea8dc71142fd8c0da))
* Reorganize all crates and provide convenience methods for dealing with parameters ([1751265](https://github.com/denehoffman/laddu/commit/1751265ba5473cad0a92eedbde6286afe60c8b92))
* **wgpu:** Reject unsupported kernel precision ([99cb507](https://github.com/denehoffman/laddu/commit/99cb50732ed46b384be7f6d7b18248f2a500a2fb))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * laddu-autodiff bumped from 0.20.0 to 0.21.0
    * laddu-compile bumped from 0.20.0 to 0.21.0
    * laddu-data bumped from 0.20.0 to 0.21.0
    * laddu-expr bumped from 0.20.0 to 0.21.0
    * laddu-kernel bumped from 0.20.0 to 0.21.0
</details>

---
This PR was generated with [Release Please](https://github.com/googleapis/release-please). See [documentation](https://github.com/googleapis/release-please#release-please).