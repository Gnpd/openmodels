# Serialized Model Format

`serialize()` turns a scikit-learn model into a plain `dict`, the same for every encoding
(`"json"`, `"pickle"`, `"msgpack"`, `"yaml"`). Format version: **4**. For what each encoding
requires you to trust, see [Security](https://github.com/Gnpd/openmodels/blob/main/SECURITY.md).

## Example

A fitted `LogisticRegression`, as JSON:

```json
{
  "estimator_class": "LogisticRegression",
  "estimator_package": "sklearn",
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
    "coef_": [[1.9208796306558977, 1.6969321460065898, -0.47739463445708574, 0.2715726107798272]],
    "intercept_": [-1.791565553734519],
    "n_features_in_": 4,
    "n_iter_": [8]
  },
  "attribute_types": {
    "classes_": "ndarray", "coef_": "ndarray", "intercept_": "ndarray",
    "n_features_in_": "int", "n_iter_": "ndarray"
  },
  "attribute_dtypes": {
    "classes_": "int64", "coef_": "float64", "intercept_": "float64", "n_iter_": "int32"
  },
  "metadata": {
    "producer_name": "openmodels",
    "producer_version": "0.2.3",
    "domain": "sklearn",
    "domain_version": "1.9.1",
    "packages": {"sklearn": "1.9.1"},
    "openmodels_format_version": 4,
    "created_at": "2026-10-08T06:19:40.587109+00:00",
    "dependency_versions": {"python": "3.11.9", "numpy": "2.4.6", "scipy": "1.17.1"}
  }
}
```

## Estimator node

Every model, and every estimator nested inside one, is a node with these fields.

| Field | Type | Meaning |
|---|---|---|
| `estimator_class` | `str` | Class name, e.g. `"LogisticRegression"`. |
| `estimator_package` | `str` | Top-level package of the class: `"sklearn"`, `"chemotools"`, `"__main__"`, ... The class is looked up by package and name among scikit-learn's and the registered custom estimators; nothing is imported. |
| `params` | `dict` | Constructor parameters (`get_params(deep=False)`). |
| `param_types` | `dict` | Type tag of each param. |
| `param_dtypes` | `dict` | dtype of each param that holds arrays. |
| `attributes` | `dict` | Fitted state: public attributes ending in `_`, plus the private ones some estimators need (`ATTRIBUTE_EXCEPTIONS`, `_n_features_out`). Absent if not fitted. |
| `attribute_types` | `dict` | Type tag of each attribute. |
| `attribute_dtypes` | `dict` | dtype of each attribute that holds arrays. |
| `metadata` | `dict` | Root node only. |

Not stored, rebuilt on load: neighbors search trees, search estimators' `scorer_`,
`HistGradientBoosting*._loss`/`_n_features` and LDA's `_max_components`.

## Type tags

A tag says how to rebuild a value; usually it's the value's type name. Tags mirror the value's
structure: a list of tags for a list, a typed entry for a dict.

| Tag | Stored as | Loads as |
|---|---|---|
| `str`, `int`, `float`, `bool`, `NoneType` | the JSON value | the same value |
| `ndarray` | nested lists; dtype text (`"float32"`, `"<U5"`, structured fields...) in the dtypes map | `np.ndarray` |
| `int8`...`uint64`, `float16`...`float64`, `bool_` | the number or boolean | that NumPy scalar |
| `[tags]` | a list | a list; tuple-only params (e.g. `feature_range`) as tuples |
| `{"tuple": [tags]}` | a list | a tuple (inside typed dicts) |
| `"dict"` | an object with string keys and plain JSON values | `dict` |
| `{"dict": {key: tag}}` | an object; dtypes map mirrors it | `dict`, each value by its tag |
| `{"dict": ..., "key_types": {key: tag}}` | non-string keys as text | `dict` with typed keys |
| `{"Bunch": {key: tag}}` | an object | `Bunch` |
| a class name, e.g. `"StandardScaler"` | an estimator node | the estimator |
| `estimators_collection` | a list (or list of lists) of nodes | a list (or 2-D object array) of estimators |
| `MaskedArray` | `{"data", "mask", "shape", "dtype"}`, plus `"types"` for object arrays | `np.ma.MaskedArray` |
| `slice` | `{"start", "stop", "step"}` | `slice` |
| `type` | `{"type_name": "float32"}` | a builtin or NumPy scalar type |
| `dtype` | the dtype's text | `np.dtype` |
| `function`, `ufunc`, `builtin_function_or_method`, `method` | `{"module", "name"}` | the function, if allowed (see below) |
| `_ArrayFunctionDispatcher` | `{"numpy_function", "module"}` | the NumPy function |
| `rv_continuous_frozen`, `rv_discrete_frozen` | `{"dist_name", "args", "kwargs"}` | frozen `scipy.stats` distribution |
| `csr_matrix` | `{"data", "indptr", "indices", "shape", "dtype"}` | `csr_matrix` (any SciPy sparse input) |
| `RandomState` | `get_state()` as a list | `np.random.RandomState` |
| `make_column_selector` | `{"pattern", "dtype_include", "dtype_exclude"}` | `make_column_selector` |
| a kernel name, e.g. `"RBF"` | `{"kernel_type", "params"}` | Gaussian-process kernel |
| a loss name, e.g. `"HalfSquaredError"` | `{"params"}` | the loss |
| `Tree` | `{"n_features", "n_outputs", "n_classes", "state", "nodes_dtype"}` | a tree's `tree_` |
| `KDTree`, `BallTree` | `{"data", "data_dtype", "sample_weight"}` | the search tree |
| `TreePredictor`, `_BisectingTree`, `_CFNode`, `_CalibratedClassifier`, `_CurveScorer`, `BSpline`, `interp1d` | the object's fields | the internal object |

Typed dict, and a dict with non-string keys:

```json
"params":       {"kw_args": {"x": [1.0, 2.0], "pair": [1, 2]}, "class_weight": {"0": 1.0, "1": 3.0}},
"param_types":  {"kw_args": {"dict": {"x": "ndarray", "pair": {"tuple": ["int", "int"]}}},
                 "class_weight": {"dict": {"0": "float", "1": "float"}, "key_types": {"0": "int", "1": "int"}}},
"param_dtypes": {"kw_args": {"x": "float32"}}
```

## dtypes

| Value | dtypes entry |
|---|---|
| array | its dtype text, e.g. `"float32"` |
| list of arrays | `""`, or one dtype per element when needed to keep it (e.g. float32 `coefs_`) |
| typed dict | a dict of the same shape with its arrays' dtypes |

## Functions

| Rule | |
|---|---|
| Saved | module and name; a bound method only if its module exposes it by name (`np.random.rand`) |
| Loaded from | already-imported modules of `numpy`, `scipy`, `sklearn`, registered estimators' packages, or `trusted_function_modules` |
| Refused | private names, `__main__`, anything that isn't a function, builtin, ufunc or bound method (`DeserializationError`) |

## Metadata

| Field | Type | Meaning |
|---|---|---|
| `producer_name` | `str` | Tool that wrote the file (`"openmodels"`). |
| `producer_version` | `str` | That tool's version. |
| `domain` | `str` | Framework the file targets: `"sklearn"`. |
| `domain_version` | `str` | scikit-learn version of the file; a different installed version warns. |
| `packages` | `dict` | `{package: version}` of every package with classes in the file; differences warn. |
| `openmodels_format_version` | `int` | `4`; a newer version than supported warns. |
| `created_at` | `str` | ISO 8601 UTC time of `serialize()`. |
| `dependency_versions` | `dict` | Writer's runtime (`python`, `numpy`, `scipy`). Not checked. |
| `title`, `description`, `license` | `str` | Optional, user-supplied. |
| `author` | `dict` | Optional: `{"name", "email"}`. |
| `metrics` | `dict` | Optional: e.g. `{"accuracy": 0.94}`. |

Optional fields are passed as `serialize(..., metadata={...})` or `save(..., metadata={...})`.

## Encodings

| Encoding | Notes |
|---|---|
| JSON | Tuples become lists. Non-finite floats are written as `NaN`/`Infinity`, which strict JSON parsers reject. |
| Pickle | The dict as is, tuples included. |
| msgpack, YAML | Tuples become lists. |

## Format versions

openmodels reads every earlier version.

| Version | Written by openmodels | Differences from 4 |
|---|---|---|
| 4 | 0.2.3 | Current. |
| 3 | 0.2.2 | No `estimator_package` (classes resolved by name); dict values untyped (`"dict"` only), non-string keys in an `__openmodels_dict__` envelope; no per-element dtypes for lists of arrays; `np.bool_` tagged `bool`. |
| 2 | 0.2.0, 0.2.1 | As 3, plus: `producer_version` is the scikit-learn version; `producers` instead of `packages`; no `domain_version`. |
| 1 | 0.1.0 and its alphas | As 2, plus: no `metadata`, its fields are top-level keys of every node. No `openmodels_format_version` (alphas) means version 1. |

openmodels 0.2.2 reads version 4 files with a warning, but loads typed dict values as plain JSON
and non-string keys as strings, and can't read per-element dtype lists.
