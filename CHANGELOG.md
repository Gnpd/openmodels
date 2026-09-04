# Changelog

All notable changes to the OpenModels project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-04

First beta. Per [Semantic Versioning](https://semver.org/spec/v2.0.0.html), any `0.y.z`
release is inherently pre-stable ("anything may change at any time"), so this is a plain
release rather than a `-beta.N` prerelease tag — `1.0.0` will mark our first public API
stability commitment.

### Changed

- **Breaking:** `SerializationManager.save()` no longer accepts an omitted `file_path`. It
  previously wrote to a default `model.{ext}` filename in the current working directory when
  none was given — unpredictable, since CWD depends on wherever the interpreter happened to be
  launched from. `file_path` is now required, matching `load()`, which already required it
- Migrated `pyproject.toml` from the deprecated `[tool.poetry]` metadata table to PEP 621's
  `[project]` table; added `keywords`, `classifiers`, and `[project.urls]` (Homepage,
  Repository, Documentation, Changelog, Issues); fixed `authors`, which was a single malformed
  string with all three names/emails comma-joined into one array entry instead of three
  separate entries
- `.github/workflows/docs.yml` now also deploys on every push to `main`, in addition to the
  existing manual `workflow_dispatch` trigger, so published docs stay in sync with `main`

### Added

- `SECURITY.md`: vulnerability reporting process, and a "Deserialization Safety" section
  distinguishing the JSON format (plain data, safe on untrusted input) from the Pickle format
  (`pickle.loads()` can execute arbitrary code — trusted sources only). The same warning was
  added to `PickleConverter`'s docstrings and the README

### Fixed

- `ClassifierChain`/`RegressorChain` construction crashed test collection entirely on
  scikit-learn 1.6.1: their wrapped-estimator constructor parameter was still named
  `base_estimator` there (renamed to `estimator` in scikit-learn 1.7.0). Now detected
  dynamically from the installed class's actual signature instead of a hardcoded name
- `SpectralEmbedding`'s `check_pipeline_consistency` conformance check flaked intermittently in
  CI: the check's default `n_neighbors` left its synthetic two-cluster dataset disconnected,
  giving the graph Laplacian's zero eigenvalue multiplicity 2 — a genuinely degenerate
  eigenspace, not just an ill-conditioned one — which ARPACK could resolve into either of two
  valid-but-different bases depending on floating-point rounding. Fixed with a per-check
  constructor override rather than changing the estimator's default construction globally
  (which broke other checks fit on much smaller synthetic data)
- mypy CI failures (untyped scipy import, protocol attribute access)

### Removed

- Stray `model.json`/`model.pkl`/`result.txt` committed at the repo root — leftover output
  from a manual `save()` call with no path (see the `save()` fix above for the root cause).
  Root-anchored `.gitignore` entries added so this can't recur

## [0.1.0-alpha.22] - 2026-08-04

### Removed

- TestPyPI publishing: dropped the `publish-test` job from `.github/workflows/ci.yml` (built a package on every `workflow_dispatch` run and published it to `test.pypi.org` if that version wasn't already there) and the now-unused `testpypi-badge.json`/README badge that displayed the last-published TestPyPI version

### Added

- `roundtrip_fit()` test helper (`openmodels/test_helpers.py`): monkeypatches `fit()`/`fit_transform()`/`fit_predict()` on given estimator classes so their fitted state is replaced with the result of an openmodels serialize→deserialize round-trip, letting any existing test suite double as a round-trip fidelity check
- `test/test_estimator_conformance.py`: runs scikit-learn's own generic `parametrize_with_checks()` battery against round-tripped instances of every estimator openmodels supports, with a strict, per-(estimator, check) xfail list distinguishing known openmodels gaps from pre-existing sklearn/check fragility
- `test/upstream/`: reuses scikit-learn's own `cross_decomposition` test suite (`test_pls.py`) unmodified against `PLSRegression`/`PLSCanonical`/`CCA`/`PLSSVD` via a `conftest.py` that registers sklearn's test fixtures as a pytest plugin
- `test/_estimator_construction.py`: shared registry of minimal constructor arguments for meta-estimators that can't be built with bare defaults (e.g. `estimator=` for `ClassifierChain`, `RFE`, `StackingRegressor`), replacing duplicated per-file special-casing across the smoke test modules
- scikit-learn 1.9.0 added to the README/docs compatibility matrix
- `test/test_serializer_base.py`, `test/test_birch_cftree.py`, `test/test_sparse_containers.py`, `test/upstream/cluster/`: regression coverage for the round-trip fixes below
- 29 stale entries removed from `test/test_estimator_conformance.py`'s `KNOWN_ROUNDTRIP_XFAILS` now that the underlying gaps are fixed

### Fixed

- `PLSRegression`, `CCA`, `PLSCanonical`, and `PLSSVD` were missing `_x_std`/`_y_mean`/`_y_std` from `ATTRIBUTE_EXCEPTIONS`, so `predict()` on a round-tripped model silently used unfitted/default scaling statistics instead of the values learned during `fit()`
- `test_others.py` estimator discovery now filters through `ALL_ESTIMATORS` so experimental-only estimators that become discoverable as a side effect of importing `sklearn.utils.estimator_checks` (e.g. `HalvingGridSearchCV`) aren't constructed without openmodels actually knowing how to serialize them
- README: corrected the scikit-learn compatibility workflow description (it runs on-demand via `workflow_dispatch`, not on every push to `main` and weekly, since push/schedule triggers were removed) and fixed a stale placeholder clone URL
- Numpy `dtype` object attributes (e.g. `SimpleImputer._fit_dtype`) failed to round-trip for any dtype other than `float64`: the type tag used to look up a deserializer was numpy's internal per-dtype subclass name (`Int64DType`, `BoolDType`, ...), which had no matching handler, so the value silently came back as a raw string instead of an `np.dtype`, and `transform()` later crashed with `'str' object has no attribute 'kind'`
- Dict-valued attributes with non-string keys (e.g. `OrdinalEncoder._missing_indices: dict[int, int]`) lost their key types on deserialize, since JSON forces string keys and there was no handler to coerce them back; `transform()` then crashed indexing with a `str` instead of an `int`. Fixed generically in `SerializerMixin` for any `int`/`float`/`bool`/`str`-keyed dict, not just this one estimator
- `BisectingKMeans`'s internal `_BisectingTree` centroids (and `KDTree` data) were silently widened from `float32` to `float64` on round-trip, since the top-level attribute-dtype tracking doesn't reach values nested inside these bespoke serializers; `predict()` then crashed with a Cython buffer dtype mismatch
- `Birch`'s fitted `root_`/`dummy_leaf_` CF-tree was reduced to an empty, structure-less stub on deserialize (only 4 scalar config values were captured, no subclusters/centroids/leaf links); `partial_fit()` on a round-tripped model then crashed trying to concatenate zero leaf centroid arrays. Now serialized and deserialized recursively, including the cross-cutting doubly-linked list of leaf nodes
- The sparse (de)serializer only recognized `scipy.sparse.csr_matrix`; `csc_matrix` and the newer array-API `csr_array`/`csc_array` containers raised `TypeError: ... is not JSON serializable` instead of round-tripping. Affected any estimator that stores its sparse training/fitted data verbatim (e.g. `KNeighborsClassifier`, `NearestNeighbors`, `DBSCAN`, `KernelRidge`)

## [0.1.0-alpha.21] - 2026-03-14

### Added

- Automated CI workflow (`.github/workflows/sklearn-compat.yml`) to test against scikit-learn 1.6.1, 1.7.2, and 1.8.0 on every push to `main`, weekly, and on demand
- README compatibility matrix listing tested scikit-learn versions with a call for users to report incompatibilities

### Fixed

- `AttributeError` when serializing `SimpleImputer` on scikit-learn < 1.8.0: `_fill_dtype` (introduced in 1.8.0) is now skipped gracefully via a `hasattr` guard, preserving compatibility across all supported versions

## [0.1.0-alpha.20] - 2025-10-01

### Added

- High-level `save()` and `load()` methods on `SerializationManager` for convenient file I/O
- README example for custom estimator support

### Changed

- Minor internal refactoring: removed redundant code and enforced UTF-8 encoding for text mode I/O

## [0.1.0-alpha.19] - 2025-06-01

### Added

- Support for custom and third-party estimators via `custom_estimators` parameter on `SklearnSerializer`
- README example showing integration with [chemotools](https://github.com/paucablop/chemotools) pipelines

## [0.1.0-alpha.16] - 2025-01-01

### Added

- [Taskfile](https://taskfile.dev/) for standardised developer workflows (`test`, `lint`, `format`, `type-check`, `build`, etc.)
- Python 3.13 added to CI matrix
- Code coverage reporting via codecov

### Changed

- Moved `SklearnSerializer` to its own subfolder (`openmodels/serializers/sklearn/`) for better organisation
- Stopped tracking `poetry.lock` in version control

## [0.1.0-alpha.14] - 2024-11-01

### Added

- Extended scikit-learn estimator support:
  - `TargetEncoder`, `SplineTransformer` (scipy BSpline), `IsolationForest`
  - `NeighborhoodComponentsAnalysis`, `LatentDirichletAllocation`
  - `ColumnTransformer`, `FeatureUnion`
  - `OutputCodeClassifier`, `OneVsOneClassifier`
  - `HDBSCAN`, `FeatureAgglomeration`, `BisectingKMeans`
  - `GenericUnivariateSelect`, `SelectFdr`, `SelectFpr`, `SelectFwe`, `SelectKBest`, `SelectPercentile`
  - `HashingVectorizer`, `FeatureHasher`, `SparseRandomProjection`, `SkewedChi2Sampler`
  - `LocalOutlierFactor` (predict-only)
- Python function serialisation support (used by feature selection estimators)

## [0.1.0-alpha.13] - 2024-10-15

### Fixed

- Dtype-robust sparse matrix comparison in tests
- `RandomTreesEmbedding` re-enabled after `OneHotEncoder` fix

## [0.1.0-alpha.12] - 2024-10-01

### Added

- Extended scikit-learn estimator support:
  - `Birch`, `TunedThresholdClassifierCV`
  - `GradientBoostingClassifier`, `GradientBoostingRegressor`
  - `HistGradientBoostingClassifier`, `HistGradientBoostingRegressor`
  - `GaussianProcessClassifier`, `GaussianProcessRegressor` (with kernel serialisation)
  - `CalibratedClassifierCV`, `LinearDiscriminantAnalysis`

## [0.1.0-alpha.11] - 2024-09-15

### Changed

- Refactored serialization layer to a mixin-based architecture (`NumpySerializerMixin`, `ScipySerializerMixin`) for extensibility and modularity
- Improved recursive deserialization for nested estimators and special types

## [0.1.0-alpha.10] - 2024-09-01

### Added

- scikit-learn version tracking: the serialized payload now records the sklearn version used, and a `UserWarning` is raised on version mismatch at deserialization time
- Dynamic TestPyPI badge in README

### Fixed

- CI badge auto-update loop

## [0.1.0-alpha.5] - 2024-08-20

### Added

- Type and dtype tracking for model parameters during serialization
- Support for nested estimators (e.g. pipelines, meta-estimators)
- `KDTree` serialization support
- `IsotonicRegression`, `TweedieRegressor`, `PoissonRegressor`, `GammaRegressor` support
- NumPy array dtype preservation (fixes `BaggingRegressor` and similar)

### Fixed

- Serialization of numpy arrays of estimators
- Pipeline serialization

## [0.1.0-alpha.4] - 2024-08-15

### Changed

- Dynamic estimator loading using `sklearn.utils.discovery.all_estimators`
- Improved attribute handling in `SklearnSerializer`

## [0.1.0-alpha.1] - 2024-08-06

### Added

- Initial release of OpenModels library
- Core functionality for serializing and deserializing machine learning models
- Support for scikit-learn models:
  - Classification: LogisticRegression, RandomForestClassifier, SVC, BernoulliNB, GaussianNB, MultinomialNB, ComplementNB, Perceptron
  - Regression: LinearRegression, Lasso, Ridge, RandomForestRegressor, SVR
  - Clustering: KMeans
  - Dimensionality Reduction: PCA
  - Other: PLSRegression
- JSON serialization format
- Pickle serialization format
- Extensible architecture for adding new model types and serialization formats
- Basic test suite for supported models
- Documentation including README, LICENSE, and CONTRIBUTING guidelines

### Security

- Implemented safe alternatives to pickle serialization

## [Unreleased]

### Planned

- Support for TensorFlow models
- YAML serialization format
- Enhanced documentation with more examples and use cases
- Support for more scikit-learn models including ensemble methods and neural networks
