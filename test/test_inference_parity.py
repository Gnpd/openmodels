"""
Post-load inference parity: for every supported estimator, every public inference method must
give the same result on the original fitted model and on its serialize -> deserialize copy.

The smoke tests (run_test_model) compare only predict/transform, and scikit-learn's conformance
battery never calls get_feature_names_out, staged_*, dbscan_clustering, ... So private state
that only those methods need could go unsaved unnoticed (see test_private_attributes.py).

`test_every_public_method_is_classified` keeps "every" true: a public method scikit-learn adds
to a supported estimator fails it until it's listed in INFERENCE_CALLS or NOT_INFERENCE.
"""

import copy
import inspect
from functools import lru_cache
from types import GeneratorType, SimpleNamespace
from typing import Any, Callable, Dict

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.base import is_regressor
from sklearn.linear_model import LogisticRegression

from openmodels.core import SerializationManager
from openmodels.serializers.sklearn.sklearn_serializer import (
    ALL_ESTIMATORS,
    NOT_SUPPORTED_ESTIMATORS,
    SklearnSerializer,
)
from test._estimator_construction import construct

pytestmark = pytest.mark.filterwarnings("ignore")

FORMATS = ["json", "pickle"]

ESTIMATOR_NAMES = sorted(
    name
    for name in ALL_ESTIMATORS
    if not name.startswith("_") and name not in NOT_SUPPORTED_ESTIMATORS
)

# ==== which methods are compared, and how they're called ====

# Inference methods: read the fitted state, leave it unchanged. Each entry calls the method
# with the estimator's data `d` (see _data_for): d.X/d.y for methods taking samples.
INFERENCE_CALLS: Dict[str, Callable[[Any, SimpleNamespace], Any]] = {
    "predict": lambda e, d: e.predict(d.X),
    "predict_proba": lambda e, d: e.predict_proba(d.X),
    "predict_log_proba": lambda e, d: e.predict_log_proba(d.X),
    "predict_joint_log_proba": lambda e, d: e.predict_joint_log_proba(d.X),
    "decision_function": lambda e, d: e.decision_function(d.X),
    "transform": lambda e, d: e.transform(d.X),
    "inverse_transform": lambda e, d: e.inverse_transform(e.transform(d.X)),
    "score": lambda e, d: e.score(d.X, d.y),
    "score_samples": lambda e, d: e.score_samples(d.X),
    "staged_predict": lambda e, d: e.staged_predict(d.X),
    "staged_predict_proba": lambda e, d: e.staged_predict_proba(d.X),
    "staged_decision_function": lambda e, d: e.staged_decision_function(d.X),
    "staged_score": lambda e, d: e.staged_score(d.X, d.y),
    "apply": lambda e, d: e.apply(d.X),
    "decision_path": lambda e, d: e.decision_path(d.X),
    "get_feature_names_out": lambda e, d: e.get_feature_names_out(),
    "get_support": lambda e, d: e.get_support(),
    "kneighbors": lambda e, d: e.kneighbors(d.X),
    "kneighbors_graph": lambda e, d: e.kneighbors_graph(d.X),
    "radius_neighbors": lambda e, d: e.radius_neighbors(d.X),
    "radius_neighbors_graph": lambda e, d: e.radius_neighbors_graph(d.X),
    "dbscan_clustering": lambda e, d: e.dbscan_clustering(0.3),
    "mahalanobis": lambda e, d: e.mahalanobis(d.X),
    "error_norm": lambda e, d: e.error_norm(np.eye(d.X.shape[1])),
    "get_covariance": lambda e, d: e.get_covariance(),
    "get_precision": lambda e, d: e.get_precision(),
    "get_depth": lambda e, d: e.get_depth(),
    "get_n_leaves": lambda e, d: e.get_n_leaves(),
    "aic": lambda e, d: e.aic(d.X),
    "bic": lambda e, d: e.bic(d.X),
    "perplexity": lambda e, d: e.perplexity(d.X),
    "reconstruction_error": lambda e, d: e.reconstruction_error(),
    "log_marginal_likelihood": lambda e, d: e.log_marginal_likelihood(),
    "latent_mean_and_variance": lambda e, d: e.latent_mean_and_variance(d.X),
    "get_indices": lambda e, d: e.get_indices(0),
    "get_shape": lambda e, d: e.get_shape(0),
    "get_submatrix": lambda e, d: e.get_submatrix(0, d.X),
    # Random, but seeded: from random_state=0 (set on every estimator that has it), from the
    # method's own random_state argument, or from the fitted random_state_, which the copy
    # restores in the same state the original has before the call.
    "sample": lambda e, d: (
        e.sample(5, random_state=0)
        if "random_state" in inspect.signature(e.sample).parameters
        else e.sample(5)
    ),
    "sample_y": lambda e, d: e.sample_y(d.X[:5]),
    "gibbs": lambda e, d: e.gibbs(d.X),
}

# Public methods that aren't inference, so aren't compared.
NOT_INFERENCE: Dict[str, str] = {
    "fit": "training",
    "fit_predict": "training",
    "fit_transform": "training",
    "partial_fit": "training",
    "path": "training (a static method computing a regularization path from data)",
    "cost_complexity_pruning_path": "training (fits a new tree from the params)",
    "get_params": "configuration",
    "set_params": "configuration",
    "set_output": "configuration",
    "set_callbacks": "configuration",
    "get_metadata_routing": "configuration",
    "set_fit_request": "configuration",
    "set_partial_fit_request": "configuration",
    "set_predict_request": "configuration",
    "set_score_request": "configuration",
    "set_transform_request": "configuration",
    "set_inverse_transform_request": "configuration",
    "densify": "changes the fitted state (coef_ storage)",
    "sparsify": "changes the fitted state (coef_ storage)",
    "restrict": "changes the fitted state (DictVectorizer vocabulary)",
    "correct_covariance": "changes the fitted state (rescales dist_; part of fit)",
    "reweight_covariance": "changes the fitted state (part of fit)",
    "build_analyzer": "depends only on params (returns a function)",
    "build_preprocessor": "depends only on params (returns a function)",
    "build_tokenizer": "depends only on params (returns a function)",
    "get_stop_words": "depends only on params",
    "decode": "depends only on params",
}

# Known, understood parity gaps: (estimator, method) -> reason. Each must still fail; once it
# passes, the test asks for the entry to be removed.
KNOWN_GAPS: Dict[tuple, str] = {
    (
        "GridSearchCV",
        "score",
    ): "scorer_ isn't saved; it should be rebuilt from `scoring` on load",
    (
        "RandomizedSearchCV",
        "score",
    ): "scorer_ isn't saved; it should be rebuilt from `scoring` on load",
}

# ==== data ====

rng = np.random.RandomState(0)
X = rng.rand(60, 4) + 0.1  # positive: needed by e.g. MultinomialNB, chi2, NMF
y = (X[:, 0] + X[:, 1] > 1.1).astype(int)
yr = X @ np.array([1.0, -2.0, 0.5, 3.0])
DOCS = [f"doc {i} word{i % 7} token{i % 3} shared" for i in range(60)]


def _data_for(name: str, estimator: Any) -> SimpleNamespace:
    """What to fit `estimator` on (fit_args) and what to call its methods with (X, y)."""
    target = yr if is_regressor(estimator) else y
    data = SimpleNamespace(X=X, y=target, fit_args=(X, target))
    if name.startswith("MultiTask") or name in (
        "MultiOutputRegressor",
        "RegressorChain",
    ):
        data.y = np.c_[yr, -yr]
    elif name in ("MultiOutputClassifier", "ClassifierChain"):
        data.y = np.c_[y, 1 - y]
    elif name in ("GammaRegressor", "PoissonRegressor"):
        data.y = yr - yr.min() + 1  # strictly positive
    elif name == "IsotonicRegression":
        data.X = X[:, 0]
    elif name == "KernelCenterer":
        data.X = X @ X.T
    elif name in ("CountVectorizer", "TfidfVectorizer", "HashingVectorizer"):
        data.X = DOCS
    elif name in ("DictVectorizer", "FeatureHasher"):
        data.X = [
            {"a": float(r[0]), "b": float(r[1]), f"c{i % 3}": 1.0}
            for i, r in enumerate(X)
        ]
    elif name in ("LabelEncoder", "LabelBinarizer"):
        data.X = y
        data.fit_args = (y,)
        return data
    elif name == "MultiLabelBinarizer":
        data.X = [{int(v), int(v) + 1} for v in y]
        data.fit_args = (data.X,)
        return data
    elif name == "PatchExtractor":
        data.X = rng.rand(3, 20, 20)  # the default patch size is a tenth of the image
    elif name in ("KNNImputer", "SimpleImputer", "MissingIndicator"):
        data.X = X.copy()
        data.X[::5, 1] = np.nan  # without NaNs, transform never reads the fitted data
    elif name == "ColumnTransformer":
        data.X = np.c_[X[:, :2], (X[:, 2] * 3).astype(int)]
    data.fit_args = (data.X, data.y)
    return data


def _construct(name: str) -> Any:
    cls = ALL_ESTIMATORS[name]
    if name == "SparseCoder":
        estimator = cls(dictionary=rng.rand(3, X.shape[1]))
    elif name == "FrozenEstimator":
        estimator = cls(LogisticRegression().fit(X, y))
    else:
        estimator = construct(cls)
    if "random_state" in estimator.get_params(deep=False):
        estimator.set_params(random_state=0)
    return estimator


@lru_cache(maxsize=None)
def _fitted(name: str) -> tuple:
    """The fitted estimator and its data, fitted once per module and never called on: each
    test calls methods on a deep copy, so seeded random state starts identical every time.
    """
    estimator = _construct(name)
    data = _data_for(name, estimator)
    try:
        estimator.fit(*data.fit_args)
    except TypeError:  # fit(X) only
        estimator.fit(data.fit_args[0])
    return estimator, data


# ==== comparison ====


def _normalize(value: Any) -> Any:
    """Turn a method's result into nested lists/arrays that can be compared directly."""
    if isinstance(value, GeneratorType):
        return [_normalize(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_normalize(v) for v in value)
    if isinstance(value, list):
        return [_normalize(v) for v in value]
    if sparse.issparse(value):
        return ("sparse", value.format, _normalize(value.toarray()))
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return (
            "pandas",
            list(getattr(value, "columns", [value.name])),
            _normalize(value.to_numpy()),
        )
    if isinstance(value, np.ndarray) and value.dtype == object:
        return [_normalize(v) for v in value.ravel()] + [("shape", value.shape)]
    return value


def _assert_same(got: Any, expected: Any) -> None:
    got, expected = _normalize(got), _normalize(expected)
    if isinstance(expected, (tuple, list)):
        assert type(got) is type(expected) and len(got) == len(
            expected
        ), f"{got!r} != {expected!r}"
        for g, e in zip(got, expected):
            _assert_same(g, e)
        return
    if isinstance(expected, (np.ndarray, np.generic)):
        got_arr, expected_arr = np.asarray(got), np.asarray(expected)
        assert (
            got_arr.dtype == expected_arr.dtype
        ), f"dtype {got_arr.dtype} != {expected_arr.dtype}"
        if np.issubdtype(expected_arr.dtype, np.inexact):
            np.testing.assert_allclose(
                got_arr, expected_arr, rtol=1e-12, atol=0, equal_nan=True
            )
        else:
            np.testing.assert_array_equal(got_arr, expected_arr)
        return
    if isinstance(expected, float):
        assert isinstance(got, float) and np.isclose(
            got, expected, rtol=1e-12, atol=0, equal_nan=True
        ), f"{got!r} != {expected!r}"
        return
    assert got == expected, f"{got!r} != {expected!r}"


def _has_method(estimator: Any, method: str) -> bool:
    # callable: some names are also params on other estimators (TSNE's perplexity is an int).
    return callable(getattr(estimator, method, None))


def _check_method(
    method: str, original: Any, loaded: Any, data: SimpleNamespace
) -> bool:
    """The loaded copy must expose the method iff the original does, raise the same type of
    error if the original raises, and otherwise return the same result. Returns whether
    results were compared (not just both missing, or both raising)."""
    has = _has_method(original, method)
    assert (
        _has_method(loaded, method) == has
    ), f"available: loaded {not has}, original {has}"
    if not has:
        return False
    call = INFERENCE_CALLS[method]
    try:
        expected = call(original, data)
        expected = _normalize(
            expected
        )  # drain generators while the original's state is current
    except Exception as e:
        with pytest.raises(type(e)):
            _normalize(call(loaded, data))
        return False
    _assert_same(call(loaded, data), expected)
    return True


# ==== tests ====


@pytest.mark.parametrize("format_name", FORMATS)
@pytest.mark.parametrize("name", ESTIMATOR_NAMES)
def test_inference_parity(name, format_name):
    pristine, data = _fitted(name)
    manager = SerializationManager(SklearnSerializer())
    loaded_pristine = manager.deserialize(
        manager.serialize(pristine, format_name), format_name
    )

    failures, stale, compared = [], [], 0
    for method in INFERENCE_CALLS:
        known = KNOWN_GAPS.get((name, method))
        # A fresh pair per method: some methods set missing private state as a side effect
        # (StackingClassifier.predict sets _n_feature_outs), which would hide a later method's
        # failure on a freshly loaded model.
        original = copy.deepcopy(pristine)
        loaded = copy.deepcopy(loaded_pristine)
        try:
            compared += _check_method(method, original, loaded, data)
        except AssertionError as e:
            if known is None:
                failures.append(f"{method}: {str(e).strip()[:300]}")
            continue
        except Exception as e:
            if known is None:
                failures.append(f"{method}: {type(e).__name__}: {str(e)[:300]}")
            continue
        if known is not None:
            stale.append(f"{method} ({known})")

    assert (
        not failures
    ), f"{name} [{format_name}] differs after loading:\n  " + "\n  ".join(failures)
    assert not stale, f"{name}: KNOWN_GAPS entries now pass, remove them: {stale}"
    # Guards against a data recipe that makes every method raise on both copies.
    has_any = any(_has_method(pristine, method) for method in INFERENCE_CALLS)
    assert (
        compared or not has_any
    ), f"{name}: no inference method returned a result to compare"


def test_every_public_method_is_classified():
    """A public method on a supported estimator that's neither compared nor listed as
    non-inference (e.g. one added by a new scikit-learn release) must be classified."""
    classified = set(INFERENCE_CALLS) | set(NOT_INFERENCE)
    unclassified = {}
    for name in ESTIMATOR_NAMES:
        cls = ALL_ESTIMATORS[name]
        for method in dir(cls):
            if method.startswith("_") or method in classified:
                continue
            if isinstance(inspect.getattr_static(cls, method), property):
                continue
            if callable(getattr(cls, method, None)):
                unclassified.setdefault(method, []).append(name)
    assert (
        not unclassified
    ), f"add these to INFERENCE_CALLS or NOT_INFERENCE: {unclassified}"


def test_known_gaps_are_real_pairs():
    for name, method in KNOWN_GAPS:
        assert name in ESTIMATOR_NAMES and method in INFERENCE_CALLS
