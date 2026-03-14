# Changelog

All notable changes to the OpenModels project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
