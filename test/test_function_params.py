"""
Functions passed as parameters must load as the identical object: NumPy/SciPy ufuncs
(`np.log1p`, `scipy.special.expit`), NumPy array functions (`np.std`, `np.linalg.norm`) and
plain functions. Builtins are outside the function allowlist unless explicitly trusted.
"""

import json

import numpy as np
import pytest
import scipy.special
from sklearn.cluster import FeatureAgglomeration
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer

from openmodels import SerializationManager, SklearnSerializer
from openmodels.exceptions import DeserializationError, SerializationError

rng = np.random.RandomState(0)
X = rng.rand(30, 4) + 0.1  # positive, so log1p/logit-friendly transforms are defined
X_UNIT = rng.rand(30, 4) * 0.8 + 0.1  # in (0, 1), for logit


def _roundtrip(model, serializer=None):
    manager = SerializationManager(serializer or SklearnSerializer())
    return manager.deserialize(manager.serialize(model))


def _transformer_roundtrip(func, data=X):
    model = FunctionTransformer(func=func).fit(data)
    loaded = _roundtrip(model)
    assert loaded.func is func
    np.testing.assert_array_equal(loaded.transform(data), model.transform(data))


# ==== ufuncs ====


@pytest.mark.parametrize("func", [np.log1p, np.exp, np.sqrt, np.abs, np.expm1])
def test_numpy_ufunc_roundtrips(func):
    _transformer_roundtrip(func)


@pytest.mark.parametrize("func", [scipy.special.expit, scipy.special.logit])
def test_scipy_ufunc_roundtrips(func):
    _transformer_roundtrip(func, X_UNIT)


@pytest.mark.parametrize(
    "func, inverse_func, y",
    [
        (np.log1p, np.expm1, X @ np.array([1.0, 2.0, 0.5, 3.0])),
        (scipy.special.logit, scipy.special.expit, X_UNIT[:, 0]),
    ],
    ids=["log1p-expm1", "logit-expit"],
)
def test_transformed_target_regressor_roundtrips(func, inverse_func, y):
    model = TransformedTargetRegressor(
        regressor=LinearRegression(), func=func, inverse_func=inverse_func
    ).fit(X, y)
    loaded = _roundtrip(model)
    assert loaded.func is func and loaded.inverse_func is inverse_func
    np.testing.assert_allclose(loaded.predict(X), model.predict(X))


def test_ufunc_written_by_earlier_versions_loads():
    # 0.2.x writes exactly this shape for np.log1p.
    serializer = SklearnSerializer()
    loaded = serializer.convert_from_serializable(
        {"module": "numpy", "name": "log1p"}, "ufunc"
    )
    assert loaded is np.log1p


def test_unlocatable_ufunc_fails_at_save_with_serialization_error():
    vectorized = np.frompyfunc(lambda a: a, 1, 1)  # no __module__, not in numpy/scipy
    with pytest.raises(SerializationError, match="Can't serialize callable"):
        SklearnSerializer().serialize(FunctionTransformer(func=vectorized))


# ==== builtins: refused unless trusted ====


def test_builtin_refused_by_default():
    with pytest.raises(DeserializationError, match="builtins.abs"):
        _roundtrip(FunctionTransformer(func=abs).fit(X))


def test_builtin_allowed_when_trusted():
    trusted = SklearnSerializer(trusted_function_modules=["builtins"])
    loaded = _roundtrip(FunctionTransformer(func=abs).fit(X), trusted)
    assert loaded.func is abs


# ==== NumPy array functions ====


@pytest.mark.parametrize(
    "func", [np.mean, np.std, np.clip, np.percentile, np.linalg.norm]
)
def test_numpy_array_function_roundtrips(func):
    model = FunctionTransformer(func=func)
    serialized = json.loads(json.dumps(SklearnSerializer().serialize(model)))
    assert SklearnSerializer().deserialize(serialized).func is func


def test_feature_agglomeration_pooling_func_roundtrips():
    model = FeatureAgglomeration(n_clusters=2, pooling_func=np.std).fit(X)
    loaded = _roundtrip(model)
    assert loaded.pooling_func is np.std
    np.testing.assert_allclose(loaded.transform(X), model.transform(X))


@pytest.mark.parametrize("name, expected", [("std", np.std), ("mean", np.mean)])
def test_array_function_written_by_earlier_versions_loads(name, expected):
    # 0.2.x wrote only the name, no module.
    serializer = SklearnSerializer()
    loaded = serializer.convert_from_serializable(
        {"numpy_function": name}, "_ArrayFunctionDispatcher"
    )
    assert loaded is expected


@pytest.mark.parametrize(
    "value",
    [
        {"numpy_function": "fft"},  # resolves to the np.fft *module*
        {"numpy_function": "_private"},
        {"numpy_function": "std", "module": "notnumpy"},
        {"numpy_function": "ndarray"},  # a class
        {"numpy_function": None},
    ],
)
def test_array_function_refused(value):
    with pytest.raises(DeserializationError, match="Unknown NumPy function"):
        SklearnSerializer().convert_from_serializable(value, "_ArrayFunctionDispatcher")
