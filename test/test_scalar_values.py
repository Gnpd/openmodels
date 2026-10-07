"""
NumPy scalar values (e.g. `Ridge(alpha=np.float32(0.5))`) must load as the same NumPy type, not
as a Python scalar. They are written as their `.item()` and tagged with their type's name.
"""

import json

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import FunctionTransformer

from openmodels import SerializationManager, SklearnSerializer

X = np.random.RandomState(0).rand(20, 3)
Y = X @ np.array([1.0, -2.0, 0.5])

SCALARS = [
    np.int8(-3),
    np.int16(-300),
    np.int32(-70000),
    np.int64(-(2**40)),
    np.uint8(200),
    np.uint16(60000),
    np.uint32(4_000_000_000),
    np.uint64(2**64 - 1),
    np.float16(0.1),
    np.float32(0.1),
    np.float64(0.1),
    np.bool_(True),
    np.bool_(False),
]


def _roundtrip(model, format_name="json"):
    manager = SerializationManager(SklearnSerializer())
    return manager.deserialize(
        manager.serialize(model, format_name=format_name), format_name=format_name
    )


def _value_roundtrip(value):
    serializer = SklearnSerializer()
    written = json.loads(json.dumps(serializer.convert_to_serializable(value)))
    return serializer.convert_from_serializable(
        written, serializer._get_nested_types(value)
    )


@pytest.mark.parametrize("value", SCALARS, ids=lambda v: f"{type(v).__name__}-{v}")
def test_scalar_roundtrips_to_same_type_and_value(value):
    loaded = _value_roundtrip(value)
    assert type(loaded) is type(value)
    assert loaded == value


@pytest.mark.parametrize("format_name", ["json", "pickle"])
def test_param_keeps_numpy_type(format_name):
    model = Ridge(alpha=np.float32(0.5)).fit(X, Y)
    loaded = _roundtrip(model, format_name)
    assert type(loaded.alpha) is np.float32
    assert loaded.alpha == model.alpha
    np.testing.assert_array_equal(loaded.predict(X), model.predict(X))


def test_scalars_inside_dict_param_keep_numpy_type():
    kw_args = {"a": np.float32(0.5), "b": np.int64(3), "c": np.bool_(True)}
    loaded = _roundtrip(FunctionTransformer(kw_args=kw_args))
    for key, value in kw_args.items():
        assert type(loaded.kw_args[key]) is type(value)
        assert loaded.kw_args[key] == value


def test_numpy_scalar_dict_keys_keep_numpy_type():
    class_weight = {np.int64(0): 1.0, np.int64(1): 3.0}
    loaded = _roundtrip(FunctionTransformer(kw_args=class_weight))
    assert [type(k) for k in loaded.kw_args] == [np.int64, np.int64]
    assert loaded.kw_args == class_weight


@pytest.mark.parametrize(
    "tag, expected",
    [
        ("int64", np.int64),
        ("float32", np.float32),
        ("bool_", np.bool_),
        ("longlong", np.longlong),
    ],
)
def test_tags_written_elsewhere_load_as_numpy_type(tag, expected):
    # "bool_" is NumPy 1's name for np.bool_; "longlong" is how Linux names np.longlong.
    loaded = SklearnSerializer().convert_from_serializable(1, tag)
    assert type(loaded) is expected
