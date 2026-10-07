# Changelog

All notable changes to the OpenModels project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.3] - 2026-10-05

### Security

- **Breaking:** functions referenced by a model file (e.g. a `SelectKBest` `score_func` or a
  `FunctionTransformer` `func`) are no longer imported from whatever module the file names. A
  function is only returned if it's a plain function, builtin or NumPy ufunc from an
  already-imported module of numpy, scipy, scikit-learn or a registered custom estimator's
  package; private names and `__main__` are refused. Anything else raises
  `DeserializationError`. Files referencing your own functions need the new
  `SklearnSerializer(trusted_function_modules=[...])`, which also allows importing those modules
- A Gaussian-process kernel's `kernel_type` read from a file must now name a `Kernel` class;
  before, any callable in `sklearn.gaussian_process.kernels` could be called with the file's
  parameters
- A SciPy distribution read from a file must now name a `scipy.stats` distribution generator;
  before, a crafted file could make loading call any public `scipy.stats` function (e.g.
  `describe`) with arguments of its choosing

### Changed

- **Wire format v4:** every estimator node records `estimator_package` (the class's top-level
  package) next to `estimator_class`, and classes are resolved by `(package, class name)`.
  Files without it (v1-v3) resolve by bare name as before, now warning when the name is
  ambiguous. Type tags are unchanged, so openmodels 0.2.2 still loads v4 files.
  `openmodels_format_version` is now `4`. See `docs/format.md`

### Fixed

- A registered custom estimator sharing a scikit-learn class's name (e.g. chemotools'
  `MinMaxScaler`) silently replaced it on load, giving a model with different output. Both now
  coexist, also inside one `Pipeline`, and so do two custom classes with the same name
- `metadata.packages` listed a custom estimator's package for models that only used the
  scikit-learn class with the same name
- An unknown estimator class raised a bare `KeyError`; it now raises
  `UnsupportedEstimatorError`. A nested unknown estimator (e.g. a `Pipeline` step) used to load
  silently as a raw dict; it now raises too
- Composite models (`Pipeline`, `FeatureUnion`, `ColumnTransformer`) saved with
  `format_name="pickle"` loaded with raw-dict steps and couldn't predict; `VotingClassifier`/
  `StackingClassifier` predicted, but their `estimators` param held raw dicts, breaking
  `clone()`. The same happened when deserializing a `serialize()` result directly. Pickle files
  written by earlier versions now load correctly
- `KNeighborsRegressor`/`RadiusNeighborsRegressor` couldn't predict after load when fitted with
  a tree (the default for dense data), and every neighbors estimator using a non-euclidean
  metric with `kd_tree`/`ball_tree` returned different neighbors after load. The search tree
  is now rebuilt on load from the estimator's own state, exactly as scikit-learn builds it;
  files written by earlier versions benefit too
- `run_test_model` test helper (`openmodels/test_helpers.py`): when sparse data was given, it
  refitted the same instance on it and round-tripped only that sparse fit, so the dense fit
  (e.g. neighbors search trees) was never tested - which hid the neighbors bug above. Dense and
  sparse fits are now round-tripped separately, each on its own clone, and the caller's model
  is no longer fitted
- NumPy types passed as parameters were loaded as Python `float`, silently changing the model:
  `CountVectorizer`'s default `dtype=np.int64` produced float64 counts after load, and e.g.
  `OneHotEncoder(dtype=np.float32)` produced float64 output. They now load as the original
  NumPy type, including from files written by earlier versions. An unknown type name still
  loads as `float`, but now with a `UserWarning`
- Functions passed as parameters: NumPy ufuncs (e.g. `FunctionTransformer(func=np.log1p)`,
  `TransformedTargetRegressor(func=np.log1p, inverse_func=np.expm1)`) loaded as raw dicts and
  failed at `transform`; SciPy ufuncs such as `scipy.special.expit`/`logit` couldn't be saved at
  all; and NumPy array functions other than mean/median/max/min/sum (e.g. `np.std`, `np.clip`,
  `np.linalg.norm`) were saved but could never be loaded. All of these now load as the original
  function, including files written by earlier versions (except submodule functions such as
  `np.linalg.norm`, whose module older files didn't record). Builtins like `abs` now raise a
  clear `DeserializationError` (they're outside the function allowlist; opt in with
  `trusted_function_modules`), and a callable that can't be located now fails at save with
  `SerializationError` instead of `AttributeError`
- SciPy distributions (e.g. in `RandomizedSearchCV(param_distributions=...)`) always loaded as
  raw dicts, and discrete ones (`randint`, `poisson`) couldn't be saved at all. Both now
  round-trip, inside `param_distributions` as a dict or a list of dicts, including continuous
  ones in files written by earlier versions
- Neighbors estimators fitted with a `BallTree` (`algorithm="ball_tree"`, or `auto` with metrics
  such as `haversine`) couldn't be saved (`TypeError`). They now save and load, including inside
  `Isomap`/`LocallyLinearEmbedding`; the tree is rebuilt on load rather than stored
- `RadiusNeighborsTransformer` didn't save its training data, so brute-force models couldn't
  `transform` after load and non-euclidean `kd_tree` ones gave wrong results
- `KernelDensity` couldn't be used after loading in any configuration: its search tree was
  restored under the wrong attribute name, without its metric or sample weights, and a
  `ball_tree` one couldn't be saved. It now round-trips exactly. Weighted models saved by earlier
  versions load with uniform weights, since their weights were never saved
- Gaussian-process kernels other than `RBF`, `WhiteKernel`, `Sum`, `Product`, `ConstantKernel`
  and `DotProduct` (e.g. `Matern`, `RationalQuadratic`, `Exponentiation`, `PairwiseKernel`)
  loaded as raw dicts when used at top level, breaking `GaussianProcessRegressor`/`Classifier`
  and `KernelRidge`; kernels with array hyperparameters (a fitted anisotropic `RBF`) or lists of
  kernels (`CompoundKernel`) couldn't be saved. All now round-trip; an unknown or abstract
  kernel type raises `DeserializationError`
- A `ColumnTransformer` couldn't `transform` pandas DataFrames after loading, so a fitted
  `Pipeline` starting with one couldn't `predict`; `get_feature_names_out` and
  `set_output(transform="pandas")` failed for any input. Its private column index map is now
  saved; files written by earlier versions need to be re-saved
- Any `slice` value (e.g. `ColumnTransformer` columns given as `slice(0, 2)`) crashed on load;
  slices inside dicts (such as `output_indices_`) now come back as slices too
- `make_column_selector` column selections can now be saved, including NumPy dtype classes
  such as `np.number`
- `SerializationManager.serialize/deserialize/save/load` now raise
  `SerializationError`/`DeserializationError` (with the original exception as the cause) for
  malformed input or unserializable models, as documented, instead of `KeyError`,
  `AttributeError` or `TypeError`. Library errors such as `UnsupportedEstimatorError` are
  unchanged
- `custom_estimators=[("Name", cls), ...]` (a list of pairs, as documented) registered nothing;
  it now works, alongside the existing forms
- Methods other than `predict`/`transform` failed on loaded models because private state wasn't
  saved:
  - `HistGradientBoostingClassifier.predict_proba` (so a soft `VotingClassifier` containing one
    couldn't `predict`), and `staged_predict*` for both HistGradientBoosting estimators;
  - `LinearDiscriminantAnalysis.transform`;
  - `get_feature_names_out` for 24 estimators, including `KMeans`, `Birch`, `Nystroem`,
    `RBFSampler`, the random projections, `PLS*`/`CCA`, `Isomap` and `Stacking*`, so a
    `Pipeline` using one of them couldn't `predict` with `set_output(transform="pandas")`;
  - `HDBSCAN.dbscan_clustering`.

  The HistGradientBoosting and LDA state is rebuilt on load, so files written by earlier
  versions work too. For `get_feature_names_out` and `HDBSCAN`, re-save older files

## [0.2.2] - 2026-09-22

### Changed

- **Wire format v3:** `metadata.producer_name`/`producer_version` now follow ONNX and name the
  tool that wrote the file - `"openmodels"` and its version for files written by
  `serialize()`, or another writer's own name/version (e.g. an R exporter). In v1/v2 they held
  the outermost model's package and, always, the scikit-learn version, which paired mismatched
  values for a third-party root model and left non-Python writers nothing honest to write.
  scikit-learn's version moved to a new `domain_version` field, which the deserialize-time
  version check now reads (falling back to `producer_version` only for v1/v2 files).
  `openmodels_format_version` is now `3`. See `docs/format.md`
- `metadata.producers` renamed `metadata.packages`, and it now always includes `sklearn`, even
  when the tree has no scikit-learn estimator class. v2 files' `producers` are still read
- The scikit-learn version check now skips an `"unknown"` version instead of warning
- `metadata.openmodels_version` removed from v3 files: `producer_version` (with
  `producer_name: "openmodels"`) records the same release. It was never read on deserialize

### Added

- Deserialize now also warns (never fails) when a non-scikit-learn package listed in
  `metadata.packages` is installed at a different version than the one recorded
- `metadata.dependency_versions` now also records the Python interpreter version (`"python"`,
  from `platform.python_version()`) - informational only, not checked on deserialize
- `docs/format.md`: "Files written by other tools" section describing how a non-openmodels
  writer should fill `metadata`

## [0.2.1] - 2026-09-14

### Fixed

- `TESTED_VERSIONS` in `sklearn_serializer.py` was missing scikit-learn 1.9.1, which had
  already been added to the README compatibility matrix and CI workflow in 0.2.0. This caused
  `SklearnSerializer` to emit a spurious "untested version" warning under scikit-learn 1.9.1
  despite it being fully tested and supported.

## [0.2.0] - 2026-09-14

### Changed

- README/docs compatibility matrix now lists scikit-learn 1.9.1 (up from 1.9.0); full test suite
  verified passing against it
- **Breaking (wire format):** `producer_version`, `producer_name`, `domain`,
  `openmodels_format_version`, and `openmodels_version` moved from flat top-level keys into a
  single `metadata` dict, present exactly once at the root of the serialized dict instead of
  being duplicated on every nested/composite sub-estimator (a `Pipeline` step, a
  `VotingClassifier`'s `estimators`, ...). `openmodels_format_version` is now `2`; files written
  with the old flat shape (version `1`) still deserialize correctly. See `docs/format.md`

### Added

- `SerializationManager.serialize()`/`.save()` now accept an optional `metadata` dict, merged
  into the serialized model's root-level `metadata` object - e.g. `title`, `description`,
  `author: {name, email}`, `license`, or free-form `metrics` - without overriding the autofilled
  producer/format fields
- New autofilled `metadata` fields: `created_at` (ISO 8601 UTC timestamp), `dependency_versions`
  (`numpy`/`scipy` versions at serialize time), and `producers` (`{package: version}` for every
  package contributing an estimator class anywhere in the tree, not just the outermost one - e.g.
  both `sklearn` and a registered third-party package for a mixed `Pipeline`)
- Two new format converters, sharing the exact same serialized dict every other format
  already used: `MsgpackConverter` (`format_name="msgpack"`) - a binary format with the same
  safe, no-code-execution data model as JSON, for large models where JSON's text encoding is
  too slow or too large - and `YAMLConverter` (`format_name="yaml"`) - for hand-editing or
  diffing a model's `metadata` cleanly, using only `yaml.safe_load`/`safe_dump`. Both require
  an optional extra (`pip install openmodels[msgpack]` / `openmodels[yaml]`) and raise a clear
  `ImportError` naming that extra if it isn't installed, rather than failing to import
  `openmodels` itself
- `FormatConverter` protocol gained a required `is_binary` class attribute, so
  `SerializationManager.load()` picks the right file open mode (text/binary) by asking the
  registered converter instead of hardcoding `format_name == "pickle"` - a latent bug that
  would have made any *other* binary format (like the new MessagePack one) unreadable via
  `load()`

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
- Full scikit-learn estimator support: `PatchExtractor` and `LocalOutlierFactor` are now
  supported, closing the last two entries in `NOT_SUPPORTED_ESTIMATORS`. Neither was a real
  serialization gap - both were artifacts of this repo's own test-harness construction/data
  choices, not of `SklearnSerializer`:
  - `LocalOutlierFactor.predict()` only exists when constructed with `novelty=True` (a
    scikit-learn API restriction, not an openmodels one - the default `novelty=False` mode is
    `fit_predict()`-only, with no `predict()` to round-trip). Its private fit-time attributes
    (`_fit_method`, `_tree`, `_fit_X`, `_distances_fit_X_`, `_lrd`) are now captured via
    `ATTRIBUTE_EXCEPTIONS`, the same pattern already used for `KNeighborsClassifier`/
    `NearestNeighbors`.
  - `PatchExtractor` is stateless (`.fit()` sets no attributes) and expects an
    `(n_images, height, width[, n_channels])` image-batch array rather than standard 2D tabular
    data; it already round-tripped correctly once given properly-shaped input.
- `openmodels_format_version`/`openmodels_version` fields on the serialized dict (see
  [issue #40](https://github.com/Gnpd/openmodels/issues/40)): the former records the wire
  format's own shape version (independent of `producer_version`, which only ever recorded
  scikit-learn's version), so future format changes have a field to check against instead of
  relying solely on ad hoc `.get(key, default)` fallbacks; the latter records the openmodels
  release that wrote the file, for tracing whether a file predates a particular bug fix. Both
  are purely additive - old files without them deserialize exactly as before, and a file with a
  `openmodels_format_version` newer than what's installed warns instead of failing outright.
  Also added `docs/format.md`, documenting the full serialized-model schema

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
