"""
Lists of arrays (e.g. MLP's `coefs_`) must keep each array's dtype. They used to be rebuilt from
their values alone, so a float32 MLP came back float64. The dtype list is only written when
that inference would be wrong, because openmodels 0.2.2 can't read one.
"""

import json

import numpy as np
import pytest
from sklearn.neural_network import MLPClassifier, MLPRegressor

from openmodels import SerializationManager, SklearnSerializer

rng = np.random.RandomState(0)
X = rng.rand(40, 3).astype(np.float32)
Y_REG = (X @ np.array([1.0, -2.0, 0.5])).astype(np.float32)
Y_CLF = (X[:, 0] > 0.5).astype(int)


def _roundtrip(model, format_name="json"):
    manager = SerializationManager(SklearnSerializer())
    return manager.deserialize(
        manager.serialize(model, format_name=format_name), format_name=format_name
    )


def _attribute_dtypes(model):
    return SklearnSerializer().serialize(model)["attribute_dtypes"]


@pytest.mark.parametrize("format_name", ["json", "pickle"])
@pytest.mark.parametrize(
    "model, y",
    [(MLPRegressor(max_iter=5), Y_REG), (MLPClassifier(max_iter=5), Y_CLF)],
    ids=["MLPRegressor", "MLPClassifier"],
)
def test_float32_mlp_keeps_float32(model, y, format_name):
    model.fit(X, y)
    loaded = _roundtrip(model, format_name)
    for name in ("coefs_", "intercepts_"):
        assert [a.dtype for a in getattr(loaded, name)] == [
            a.dtype for a in getattr(model, name)
        ]
        for original, restored in zip(getattr(model, name), getattr(loaded, name)):
            np.testing.assert_array_equal(restored, original)
    method = "predict_proba" if hasattr(model, "predict_proba") else "predict"
    expected = getattr(model, method)(X)
    result = getattr(loaded, method)(X)
    assert result.dtype == expected.dtype == np.float32
    np.testing.assert_array_equal(result, expected)


def test_float64_mlp_writes_no_dtype_list():
    # What 0.2.2 can read: lists of arrays their values rebuild exactly get no dtype list.
    model = MLPRegressor(max_iter=5).fit(X.astype(np.float64), Y_REG.astype(np.float64))
    dtypes = _attribute_dtypes(model)
    assert dtypes["coefs_"] == ""
    assert dtypes["intercepts_"] == ""


@pytest.mark.parametrize(
    "arrays",
    [
        [np.zeros(2, np.float32), np.ones(3, np.float64)],
        [np.arange(3, dtype=np.uint8), np.arange(2, dtype=np.int64)],
        [np.arange(3, dtype=np.int16), np.array([True, False])],
        [np.zeros(0, np.int64), np.zeros(2, np.float32)],
    ],
    ids=["float32-float64", "uint8-int64", "int16-bool", "empty-int64"],
)
def test_mixed_list_keeps_every_dtype(arrays):
    serializer = SklearnSerializer()
    written = json.loads(json.dumps(serializer.convert_to_serializable(arrays)))
    loaded = serializer.convert_from_serializable(
        written, serializer._get_nested_types(arrays), serializer._get_dtype(arrays)
    )
    assert [a.dtype for a in loaded] == [a.dtype for a in arrays]
    for original, restored in zip(arrays, loaded):
        np.testing.assert_array_equal(restored, original)


def test_file_without_dtype_list_loads_as_before():
    # Files written before the dtype list existed: arrays are rebuilt from their values.
    model = MLPRegressor(max_iter=5).fit(X, Y_REG)
    data = SklearnSerializer().serialize(model)
    data["attribute_dtypes"]["coefs_"] = ""
    data["attribute_dtypes"]["intercepts_"] = ""
    loaded = SklearnSerializer().deserialize(data)
    assert [a.dtype for a in loaded.coefs_] == [np.float64, np.float64]
    np.testing.assert_allclose(loaded.predict(X), model.predict(X), rtol=1e-5)
