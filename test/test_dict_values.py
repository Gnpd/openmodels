"""
Values inside dicts (FunctionTransformer's kw_args, a search's cv_results_, Voting*/Stacking*'s
named_estimators_, ...): formats save them as plain JSON, so arrays, tuples, estimators and the
like are typed per key ({"dict": {key: type}}) and restored on load. Non-string keys (e.g.
class_weight={0: 1.0}) are saved as text, with their types in "key_types". Dicts of plain JSON
values with string keys keep the "dict" tag, and files without per-key types still load as
before.
"""

import json

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.utils import Bunch

from openmodels.core import SerializationManager
from openmodels.exceptions import SerializationError
from openmodels.serializers.sklearn.sklearn_serializer import SklearnSerializer

FORMATS = ["json", "pickle"]

rng = np.random.RandomState(0)
X = rng.rand(40, 3)
y = np.arange(40) % 2


def _roundtrip(model, format_name="json"):
    manager = SerializationManager(SklearnSerializer())
    return manager.deserialize(manager.serialize(model, format_name), format_name)


def _assert_same(restored, original, path="value"):
    """Same value and same types all the way down (a list is not a tuple, a list is not an
    array, a float64 array is not a float32 one)."""
    assert type(restored) is type(
        original
    ), f"{path}: {type(restored)} != {type(original)}"
    if isinstance(original, np.ma.MaskedArray):
        assert restored.dtype == original.dtype, path
        np.testing.assert_array_equal(restored.mask, original.mask)
        for i in np.flatnonzero(~np.ma.getmaskarray(original)):
            _assert_same(restored.data[i], original.data[i], f"{path}[{i}]")
    elif isinstance(original, np.ndarray):
        assert restored.dtype == original.dtype, path
        np.testing.assert_array_equal(restored, original)
    elif isinstance(original, dict):
        assert list(restored) == list(original), path
        assert [type(k) for k in restored] == [type(k) for k in original], path
        for key in original:
            _assert_same(restored[key], original[key], f"{path}[{key!r}]")
    elif isinstance(original, (list, tuple)):
        assert len(restored) == len(original), path
        for i, (r, o) in enumerate(zip(restored, original)):
            _assert_same(r, o, f"{path}[{i}]")
    elif hasattr(original, "get_params"):
        assert restored.get_params() == original.get_params(), path
    else:
        assert restored == original, path


KW_ARGS = {
    "array": np.array([1.0, 2.0], dtype=np.float32),
    "int_array": np.arange(3, dtype=np.int16),
    "pair": (1, 2),
    "pairs": [(1, "a"), (2, "b")],
    "nested": {"inner": np.zeros((2, 2)), "plain": [1, 2]},
    "cols": slice(1, 3),
    "scalar": np.float64(0.5),
    "int_keys": {0: 1.0, 1: 3.0},
    "plain": "x",
    "none": None,
}


@pytest.mark.parametrize("format_name", FORMATS)
def test_dict_values_restored(format_name):
    loaded = _roundtrip(FunctionTransformer(kw_args=KW_ARGS), format_name)
    _assert_same(loaded.kw_args, KW_ARGS, "kw_args")


@pytest.mark.parametrize("format_name", FORMATS)
def test_estimator_inside_dict_restored(format_name):
    kw_args = {"scaler": StandardScaler(with_mean=False)}
    loaded = _roundtrip(FunctionTransformer(kw_args=kw_args), format_name)
    _assert_same(loaded.kw_args, kw_args, "kw_args")


def test_typed_dict_format():
    """How a dict that needs restoring is written: data unchanged, per-key types next to it,
    and dtypes for its arrays."""
    model = FunctionTransformer(
        kw_args={"x": np.array([1.0, 2.0], dtype=np.float32), "pair": (1, 2), "n": 3}
    )
    data = json.loads(json.dumps(SklearnSerializer().serialize(model)))
    assert data["params"]["kw_args"] == {"x": [1.0, 2.0], "pair": [1, 2], "n": 3}
    assert data["param_types"]["kw_args"] == {
        "dict": {"x": "ndarray", "pair": {"tuple": ["int", "int"]}, "n": "int"}
    }
    assert data["param_dtypes"]["kw_args"] == {"x": "float32"}


def test_plain_dict_keeps_dict_tag():
    kw_args = {"a": 1, "b": [1.0, 2.0], "c": None, "d": "x", "e": True}
    data = SklearnSerializer().serialize(FunctionTransformer(kw_args=kw_args))
    assert data["param_types"]["kw_args"] == "dict"
    assert "kw_args" not in data["param_dtypes"]
    assert _roundtrip(FunctionTransformer(kw_args=kw_args)).kw_args == kw_args


def test_files_without_dict_types_still_load():
    """Files written before dicts were typed per key tag them "dict": values load as their
    saved JSON, except slices, which are recognised by shape."""
    data = SklearnSerializer().serialize(
        FunctionTransformer(kw_args={"cols": slice(1, 3), "x": np.zeros(2)})
    )
    data["param_types"]["kw_args"] = "dict"
    data["param_dtypes"].pop("kw_args", None)
    loaded = SerializationManager(SklearnSerializer()).deserialize(json.dumps(data))
    assert loaded.kw_args == {"cols": slice(1, 3), "x": [0.0, 0.0]}


@pytest.mark.parametrize("format_name", FORMATS)
def test_class_weight_keys_restored_and_refittable(format_name):
    model = LogisticRegression(class_weight={0: 1.0, 1: 3.0}).fit(X, y)
    loaded = _roundtrip(model, format_name)
    _assert_same(loaded.class_weight, model.class_weight, "class_weight")
    loaded.fit(X, y)
    clone(loaded).fit(X, y)


def test_non_string_keys_format():
    """Non-string keys are written as text, with their types next to the value types; no
    keys/values envelope."""
    model = LogisticRegression(class_weight={0: 1.0, 1: 3.0})
    data = SklearnSerializer().serialize(model)
    assert data["params"]["class_weight"] == {"0": 1.0, "1": 3.0}
    assert data["param_types"]["class_weight"] == {
        "dict": {"0": "float", "1": "float"},
        "key_types": {"0": "int", "1": "int"},
    }
    assert "__openmodels_dict__" not in json.dumps(data)


MIXED_KEYS = {
    0: np.array([1.0, 2.0], dtype=np.float32),
    1.5: (1, 2),
    True: "t",
    None: {3: "x"},
    "a": StandardScaler(),
}


@pytest.mark.parametrize("format_name", FORMATS)
def test_mixed_keys_restored(format_name):
    loaded = _roundtrip(FunctionTransformer(kw_args=MIXED_KEYS), format_name)
    _assert_same(loaded.kw_args, MIXED_KEYS, "kw_args")


def test_numpy_keys_restored():
    """NumPy scalar keys (e.g. class labels from np.unique) load with equal values."""
    kw_args = {np.int64(7): 1, np.float32(0.5): 2}
    loaded = _roundtrip(FunctionTransformer(kw_args=kw_args))
    assert loaded.kw_args == {7: 1, 0.5: 2}


@pytest.mark.parametrize(
    "kw_args",
    [{1: "a", "1": "b"}, {(1, 2): "a"}],
    ids=["keys-with-same-text", "tuple-key"],
)
def test_unsupported_keys_refused_at_save(kw_args):
    manager = SerializationManager(SklearnSerializer())
    with pytest.raises(SerializationError):
        manager.serialize(FunctionTransformer(kw_args=kw_args))


def test_files_with_envelope_dicts_still_load():
    """Files written before format v4 saved dicts with non-string keys as a keys/values
    envelope tagged "dict"."""
    data = SklearnSerializer().serialize(LogisticRegression())
    data["params"]["class_weight"] = {
        "__openmodels_dict__": True,
        "keys": [0, 1],
        "key_types": ["int", "int"],
        "values": [1.0, 3.0],
    }
    data["param_types"]["class_weight"] = "dict"
    loaded = SerializationManager(SklearnSerializer()).deserialize(json.dumps(data))
    _assert_same(loaded.class_weight, {0: 1.0, 1: 3.0}, "class_weight")


SEARCHES = {
    # param_* columns are masked arrays: float, string and object (None/dict) ones.
    "grid": GridSearchCV(
        SVC(),
        [
            {"C": [0.1, 1.0], "kernel": ["rbf"]},
            {"kernel": ["linear"], "class_weight": [None, {0: 1, 1: 2}]},
        ],
        cv=2,
    ),
    "random": RandomizedSearchCV(
        LogisticRegression(),
        {"C": np.array([0.1, 1.0, 10.0])},
        n_iter=2,
        cv=2,
        random_state=0,
    ),
}


@pytest.mark.parametrize("format_name", FORMATS)
@pytest.mark.parametrize("name", SEARCHES)
def test_cv_results_restored(name, format_name):
    search = clone(SEARCHES[name]).fit(X, y)
    loaded = _roundtrip(search, format_name)
    _assert_same(loaded.cv_results_, search.cv_results_, "cv_results_")
    _assert_same(loaded.best_params_, search.best_params_, "best_params_")


@pytest.mark.parametrize("format_name", FORMATS)
@pytest.mark.parametrize("cls", [VotingClassifier, StackingClassifier])
def test_named_estimators_restored(cls, format_name):
    model = cls([("lr", LogisticRegression()), ("svc", SVC())]).fit(X, y)
    loaded = _roundtrip(model, format_name)
    assert isinstance(loaded.named_estimators_, Bunch)
    _assert_same(
        dict(loaded.named_estimators_),
        dict(model.named_estimators_),
        "named_estimators_",
    )
    np.testing.assert_array_equal(
        loaded.named_estimators_.lr.predict(X), model.named_estimators_.lr.predict(X)
    )
