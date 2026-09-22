# Serialized Model Format

`SerializationManager.serialize()`/`SklearnSerializer.serialize()` produce a plain Python
`dict`. This page documents that dict's shape - every `FormatConverter` (`JSONConverter`,
`PickleConverter`, `MsgpackConverter`, `YAMLConverter`) encodes the exact same dict, just in a
different wire encoding (`format_name="json"`/`"pickle"`/`"msgpack"`/`"yaml"`); see
[Security](https://github.com/Gnpd/openmodels/blob/main/SECURITY.md) for each format's trust
requirements, and `README.md`'s Security section for the short version.

## Top-level fields

| Field | Type | Meaning |
|---|---|---|
| `estimator_class` | `str` | The model's class name, e.g. `"LogisticRegression"`. Looked up against a registry of known scikit-learn (and registered custom) classes on deserialize. |
| `params` | `dict` | The estimator's constructor parameters, from `model.get_params(deep=False)`. |
| `param_types` | `dict` | `{param_name: type(value).__name__}` for every entry in `params` - lets the deserializer reconstruct types JSON can't represent natively (e.g. `tuple`). |
| `param_dtypes` | `dict` | Like `param_types`, but for numpy dtypes specifically (e.g. a `np.dtype` param), when applicable. |
| `attributes` | `dict` | The model's *fitted* state - every public attribute following scikit-learn's trailing-underscore convention (`coef_`, `classes_`, ...), plus a small per-class allowlist of private attributes some estimators need at predict/transform time (see `ATTRIBUTE_EXCEPTIONS` in `sklearn_serializer.py`). Omitted entirely if the model hasn't been fit yet. |
| `attribute_types` | `dict` | Like `param_types`, for `attributes`. |
| `attribute_dtypes` | `dict` | Like `param_dtypes`, for `attributes` - this is how e.g. a `float32` array survives the round trip instead of silently widening to JSON's one numeric type. |
| `metadata` | `dict` | Producer/format bookkeeping plus optional user-supplied fields. Present exactly once, only at the root of the serialized dict - see below. Absent on nested/composite sub-estimators. |

## `metadata`

The root-level `metadata` object mixes two kinds of fields:

**Autofilled** (written by the model serializer, e.g. `SklearnSerializer`, on every call to
`serialize()`):

| Field | Type | Meaning |
|---|---|---|
| `producer_name` | `str` | The top-level package the *outermost* model class belongs to (`model.__module__.split(".")[0]`) - `"sklearn"` for any scikit-learn estimator, something else (e.g. `"chemotools"`) for a registered third-party one. A class defined in a script or notebook reports its module name instead (e.g. `"__main__"`). |
| `producer_version` | `str` | The version of the `producer_name` package - always equal to `producers[producer_name]`. For a scikit-learn root model this is the scikit-learn version; for a third-party root model it's *that* package's version, not scikit-learn's. `"unknown"` if it can't be resolved (see `producers`). |
| `producers` | `dict` | `{package_name: version}` for every package contributing an estimator class anywhere in the tree - not just the outermost one - plus the `domain` package (`sklearn`), which is always listed. For a plain scikit-learn model this is the single entry `{"sklearn": <version>}`; for a third-party root model or e.g. a `Pipeline` mixing a scikit-learn step with a registered third-party step, it has one entry per distinct package. Version lookup is best-effort (an already-imported module's own `__version__` attribute, falling back to installed-package metadata, falling back to `"unknown"`) since it must work for packages like scikit-learn itself, whose import name (`sklearn`) differs from its distribution name (`scikit-learn`). On deserialize, every non-`sklearn` entry is compared with the installed version and a mismatch raises a `UserWarning` (entries that are `"unknown"` on either side are skipped). |
| `domain` | `str` | The framework the model serializer targets. Currently always `"sklearn"`. Reserved for future non-scikit-learn model serializers. |
| `domain_version` | `str` | The `domain` framework's version at serialize time (`sklearn.__version__`), independent of which package the outermost model comes from. This is the version the deserialize-time check compares against the installed scikit-learn: any difference in the version string (patch releases included) raises a `UserWarning` listing the tested versions - it never blocks loading. Files older than format `3` have no `domain_version`; for those, `producer_version` (which then always held the scikit-learn version) is used instead. |
| `openmodels_format_version` | `int` | The version of *this wire format's shape*, independent of both `domain_version` and `openmodels_version` - bumped only when the structure or meaning of the fields documented on this page changes. Currently `3`. A file with no `openmodels_format_version` key predates this field and has version `1`'s flat shape (see "Format history" below); a value newer than what your installed openmodels understands triggers a `UserWarning` (not an error) on deserialize. |
| `openmodels_version` | `str` | The openmodels release that wrote this file (from package metadata). Informational only - not checked at deserialize time. Useful for tracing whether a file was written before a particular bug fix landed. |
| `created_at` | `str` | ISO 8601 UTC timestamp of when `serialize()` was called. Informational only - not checked at deserialize time, and makes two serializations of an otherwise-identical model no longer byte-identical. |
| `dependency_versions` | `dict` | `{"python": ..., "numpy": ..., "scipy": ...}` - the runtime environment at serialize time: the Python interpreter version (`platform.python_version()`, e.g. `"3.11.9"`) plus the array/sparse-matrix libraries this format's dtype-preserving round trip depends on. Informational only (not checked) - the serialized dict itself is Python-version-independent; these are for tracing and reproducing the environment that produced a file. |

**Optional** (user-supplied, passed via `SerializationManager.serialize()`/`.save()`'s
`metadata` parameter and merged into this same object - an autofilled field above always wins
if a key collides):

| Field | Type | Meaning |
|---|---|---|
| `title` | `str` | A short human-readable name for the model. |
| `description` | `str` | A longer free-text description. |
| `author` | `dict` | `{"name": str, "email": str}`. |
| `license` | `str` | The model's license, e.g. `"MIT"` or `"CC-BY-4.0"`. |
| `metrics` | `dict` | Free-form evaluation metrics, e.g. `{"accuracy": 0.94}` - openmodels doesn't give this a fixed schema, since it's evaluation metadata rather than something the serializer itself produces or checks. |

```python
manager.save(
    model,
    "model.json",
    metadata={
        "title": "Customer churn classifier",
        "description": "Trained on Q3 2026 data.",
        "author": {"name": "Jane Doe", "email": "jane@example.com"},
        "license": "MIT",
        "metrics": {"accuracy": 0.94, "f1": 0.91},
    },
)
```

## Example

A fitted `LogisticRegression`, serialized to JSON:

```json
{
  "estimator_class": "LogisticRegression",
  "params": {
    "C": 1.0, "class_weight": null, "dual": false, "fit_intercept": true,
    "intercept_scaling": 1, "l1_ratio": 0.0, "max_iter": 100, "n_jobs": null,
    "penalty": "deprecated", "random_state": null, "solver": "lbfgs",
    "tol": 0.0001, "verbose": 0, "warm_start": false
  },
  "param_types": {
    "C": "float", "class_weight": "NoneType", "dual": "bool", "fit_intercept": "bool",
    "intercept_scaling": "int", "l1_ratio": "float", "max_iter": "int", "n_jobs": "NoneType",
    "penalty": "str", "random_state": "NoneType", "solver": "str", "tol": "float",
    "verbose": "int", "warm_start": "bool"
  },
  "param_dtypes": {},
  "attributes": {
    "classes_": [0, 1],
    "coef_": [[1.88878277495458, -0.6412359253840524, 0.2560199894923986, 0.0321823421317183]],
    "intercept_": [0.6645191069666503]
  },
  "attribute_types": {
    "classes_": "ndarray", "coef_": "ndarray", "intercept_": "ndarray"
  },
  "attribute_dtypes": {
    "classes_": "int64", "coef_": "float64", "intercept_": "float64"
  },
  "metadata": {
    "producer_version": "1.9.0",
    "producer_name": "sklearn",
    "producers": {"sklearn": "1.9.0"},
    "domain": "sklearn",
    "domain_version": "1.9.0",
    "openmodels_format_version": 3,
    "openmodels_version": "0.2.0",
    "created_at": "2026-09-05T12:00:00+00:00",
    "dependency_versions": {"python": "3.11.9", "numpy": "2.1.0", "scipy": "1.14.0"},
    "title": "Customer churn classifier"
  }
}
```

When the outermost model comes from a registered third-party package, the `producer_*` pair
describes that package while `domain_*` still records scikit-learn:

```json
"metadata": {
  "producer_name": "chemotools",
  "producer_version": "0.4.4",
  "producers": {"chemotools": "0.4.4", "sklearn": "1.9.0"},
  "domain": "sklearn",
  "domain_version": "1.9.0",
  ...
}
```

## Nested and composite estimators

A meta-estimator that holds other estimators - a `Pipeline` step, a `VotingClassifier`'s
`estimators`, a `ColumnTransformer`'s `transformers` - doesn't get a special graph
representation. Wherever a `BaseEstimator` value appears (inside `params` or `attributes`), it's
serialized recursively as a nested copy of this same dict shape, minus `metadata` - that block is
root-only, so it's never duplicated across a pipeline's steps. A serialized `Pipeline`
is a normal JSON tree, not a separate node/edge graph - openmodels reconstructs models by
calling `set_params()`/rebuilding fitted attributes on the real scikit-learn class, not by
executing an independent computation graph, so there's no separate graph representation to keep
in sync with it.

## Format history

- **v1** (initial): `producer_version`, `producer_name`, `domain`, `openmodels_format_version`,
  `openmodels_version` were flat top-level keys, duplicated on every nested/composite
  sub-estimator. A file with no `openmodels_format_version` key at all is version `1`.
- **v2**: those same fields moved into a single `metadata` dict, present once at the root;
  optional user metadata (`title`/`description`/`author`/`license`/`metrics`) and additional
  autofilled fields (`producers`, `created_at`, `dependency_versions`) added. New fields inside
  `metadata` don't need their own format-version bump - only a change to the structure itself
  does - so a `metadata` dict missing one of these (e.g. a file written by an earlier v2 build)
  still deserializes fine; code reading these fields should use their absence gracefully rather
  than assume every v2 file has all of them.
- **v3**: `producer_version` now records the version of the `producer_name` package (the
  outermost model's) - in v1/v2 it was always the scikit-learn version, even when
  `producer_name` named a third-party package. The scikit-learn version moved to the new
  `domain_version` field, and `producers` always includes `sklearn`. Readers use
  `domain_version` when present and fall back to `producer_version` for v1/v2 files.

## Why not ONNX or PMML?

Formats like [ONNX](https://onnx.ai/) exist to describe an operator computation graph that a
*separate, cross-language runtime* can execute without the original framework. openmodels never
executes a model itself - it always reconstructs the real scikit-learn object and lets
scikit-learn run it - so there's no independent graph to encode, and adopting one would add
significant complexity without a corresponding benefit for this library's actual use case.
