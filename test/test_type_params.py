"""
Type-valued params (e.g. `dtype=np.int64`) must load as the identical type. They are written as
the type's `__name__`; NumPy scalar types used to fall back to Python `float`.
"""

import json
import warnings

import numpy as np
import pytest
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.feature_extraction.text import (
    CountVectorizer,
    HashingVectorizer,
    TfidfVectorizer,
)
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from openmodels import SerializationManager, SklearnSerializer

DOCS = ["apple banana apple", "banana cherry", "cherry date apple"]
CATEGORIES = [["a"], ["b"], ["c"], ["a"]]


def _roundtrip(model):
    manager = SerializationManager(SklearnSerializer())
    return manager.deserialize(manager.serialize(model))


def _type_roundtrip(t):
    serializer = SklearnSerializer()
    written = json.loads(json.dumps(serializer.convert_to_serializable(t)))
    return serializer.convert_from_serializable(
        written, serializer._get_nested_types(t)
    )


@pytest.mark.parametrize(
    "t",
    [
        int,
        float,
        bool,
        str,
        bytes,
        complex,
        object,
        np.int64,
        np.int32,
        np.uint8,
        np.float16,
        np.float32,
        np.float64,
        np.complex128,
        np.str_,
        np.datetime64,
    ],
)
def test_type_roundtrips_to_identical_object(t):
    assert _type_roundtrip(t) is t


def test_numpy_bool_loads_as_python_bool():
    # Both are named "bool", so the file can't tell them apart; they're the same dtype.
    assert _type_roundtrip(np.bool_) is bool
    assert np.dtype(bool) == np.dtype(np.bool_)


@pytest.mark.parametrize(
    "model, data",
    [
        (CountVectorizer(), DOCS),
        (TfidfVectorizer(), DOCS),
        (HashingVectorizer(n_features=16), DOCS),
        (DictVectorizer(), [{"a": 1}, {"b": 2}]),
        (FeatureHasher(n_features=8), [{"a": 1}, {"b": 2}]),
        (OneHotEncoder(), CATEGORIES),
        (OrdinalEncoder(), CATEGORIES),
    ],
    ids=lambda v: type(v).__name__ if not isinstance(v, list) else "",
)
def test_default_dtype_estimators_keep_their_dtype(model, data):
    model.fit(data)
    assert _roundtrip(model).dtype is model.dtype


@pytest.mark.parametrize(
    "model, data, expected",
    [
        (CountVectorizer(), DOCS, np.int64),
        (TfidfVectorizer(dtype=np.float32), DOCS, np.float32),
        (OneHotEncoder(dtype=np.float32), CATEGORIES, np.float32),
    ],
    ids=["CountVectorizer", "TfidfVectorizer-float32", "OneHotEncoder-float32"],
)
def test_output_dtype_preserved(model, data, expected):
    model.fit(data)
    loaded = _roundtrip(model)
    assert loaded.transform(data).dtype == expected
    assert loaded.transform(data).dtype == model.transform(data).dtype


def test_file_written_by_earlier_versions_loads_numpy_type():
    # 0.2.x writes exactly this shape.
    serializer = SklearnSerializer()
    assert serializer.convert_from_serializable({"type_name": "int64"}, "type") is (
        np.int64
    )


@pytest.mark.parametrize("name", ["nonsense", "f8", "i4,f8", "void", None])
def test_unknown_or_non_canonical_name_warns_and_loads_as_float(name):
    serializer = SklearnSerializer()
    with pytest.warns(UserWarning, match="Unknown type"):
        result = serializer.convert_from_serializable({"type_name": name}, "type")
    assert result is float


def test_known_names_dont_warn():
    serializer = SklearnSerializer()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for name in ["int", "float64", "int64", "float32"]:
            serializer.convert_from_serializable({"type_name": name}, "type")
