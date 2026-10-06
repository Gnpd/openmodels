"""
Format v4: estimator classes are identified by (package, class name), and function/kernel
references read from a file are restricted to an allowlist.
"""

import copy
import json
import sys
import warnings

import numpy as np
import pytest
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, StandardScaler

from openmodels.core import SerializationManager
from openmodels.exceptions import DeserializationError, UnsupportedEstimatorError
from openmodels.serializers.sklearn.sklearn_serializer import SklearnSerializer

X = np.array([[1.0, 5.0, 2.0], [4.0, 0.0, 3.0], [2.0, 2.0, 8.0], [0.0, 1.0, 1.0]])
y = np.array([0, 1, 1, 0])


def _make_scaler(package: str, offset: float) -> type:
    """A class named MinMaxScaler that lives in `package` and scales each row (not each
    column) like chemotools' MinMaxScaler, so its output differs from scikit-learn's."""

    class MinMaxScaler(TransformerMixin, BaseEstimator):
        def __init__(self, use_min=True):
            self.use_min = use_min

        def fit(self, X, y=None):
            self.offset_ = offset
            self.n_features_in_ = np.asarray(X).shape[1]
            return self

        def transform(self, X):
            X = np.asarray(X, dtype=float)
            return X / X.max(axis=1, keepdims=True) + self.offset_

    MinMaxScaler.__module__ = f"{package}.scale"
    return MinMaxScaler


FakeMinMaxScaler = _make_scaler("fakepkg", 0.0)
OtherMinMaxScaler = _make_scaler("otherpkg", 10.0)


def _serializer(*classes, **kwargs):
    # A list of dicts, so classes sharing a name stay distinct registrations.
    return SklearnSerializer(
        custom_estimators=[{cls.__name__: cls} for cls in classes], **kwargs
    )


def _json(serialized):
    # Like a real file: tuples in the type maps become lists, which deserialization expects.
    return json.loads(json.dumps(serialized))


def _roundtrip(serializer, model):
    return serializer.deserialize(_json(serializer.serialize(model)))


def _estimator_nodes(obj):
    """Every serialized estimator node (dict with "estimator_class") in obj."""
    if isinstance(obj, dict):
        if "estimator_class" in obj:
            yield obj
        for value in obj.values():
            yield from _estimator_nodes(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _estimator_nodes(item)


def _as_v3(serialized):
    """A v4 dict turned into what format v3 wrote: no estimator_package anywhere."""
    old = _json(serialized)
    for node in _estimator_nodes(old):
        node.pop("estimator_package", None)
    old["metadata"]["openmodels_format_version"] = 3
    return old


def _ambiguity_warnings(caught):
    return [w for w in caught if "resolved to" in str(w.message)]


# ==== Identity written ====


def test_every_node_records_its_package():
    serializer = _serializer(FakeMinMaxScaler)
    pipeline = Pipeline([("fake", FakeMinMaxScaler()), ("sk", MinMaxScaler())]).fit(X)
    serialized = serializer.serialize(pipeline)

    nodes = list(_estimator_nodes(serialized))
    assert [(n["estimator_class"], n["estimator_package"]) for n in nodes] == [
        ("Pipeline", "sklearn"),
        ("MinMaxScaler", "fakepkg"),
        ("MinMaxScaler", "sklearn"),
    ]
    assert "estimator_package" not in serialized["metadata"]


def test_metadata_packages_not_mislabelled_by_colliding_custom_class():
    serializer = _serializer(FakeMinMaxScaler)
    packages = serializer.serialize(MinMaxScaler().fit(X))["metadata"]["packages"]
    assert packages == {"sklearn": sklearn.__version__}


# ==== Collision coexistence ====


@pytest.mark.parametrize("cls", [MinMaxScaler, FakeMinMaxScaler])
def test_same_named_classes_each_resolve_to_their_own(cls):
    model = cls().fit(X)
    back = _roundtrip(_serializer(FakeMinMaxScaler), model)
    assert type(back) is cls
    np.testing.assert_array_equal(back.transform(X), model.transform(X))


def test_same_named_classes_coexist_in_one_pipeline():
    pipeline = Pipeline([("fake", FakeMinMaxScaler()), ("sk", MinMaxScaler())]).fit(X)
    back = _roundtrip(_serializer(FakeMinMaxScaler), pipeline)
    assert type(back.steps[0][1]) is FakeMinMaxScaler
    assert type(back.steps[1][1]) is MinMaxScaler
    np.testing.assert_array_equal(back.transform(X), pipeline.transform(X))


def test_two_custom_classes_with_the_same_name_coexist():
    serializer = _serializer(FakeMinMaxScaler, OtherMinMaxScaler)
    pipeline = Pipeline(
        [("fake", FakeMinMaxScaler()), ("other", OtherMinMaxScaler())]
    ).fit(X)
    back = _roundtrip(serializer, pipeline)
    assert type(back.steps[0][1]) is FakeMinMaxScaler
    assert type(back.steps[1][1]) is OtherMinMaxScaler
    np.testing.assert_array_equal(back.transform(X), pipeline.transform(X))


# ==== Files without estimator_package (v1-v3) ====


def test_old_file_resolves_by_bare_name_without_warning():
    serializer = SklearnSerializer()
    old = _as_v3(serializer.serialize(MinMaxScaler().fit(X)))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        back = serializer.deserialize(old)
    assert type(back) is MinMaxScaler
    assert not _ambiguity_warnings(caught)


def test_old_file_keeps_custom_wins_rule_and_warns_once():
    pipeline = Pipeline([("a", MinMaxScaler()), ("b", MinMaxScaler())]).fit(X)
    old = _as_v3(SklearnSerializer().serialize(pipeline))

    serializer = _serializer(FakeMinMaxScaler)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        back = serializer.deserialize(old)
    assert all(type(step) is FakeMinMaxScaler for _, step in back.steps)
    assert len(_ambiguity_warnings(caught)) == 1

    # Reset per deserialize() call, so the next load warns again.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        serializer.deserialize(old)
    assert len(_ambiguity_warnings(caught)) == 1


# ==== Unregistered / unknown classes ====


def test_unregistered_package_raises_and_imports_nothing():
    data = {
        "estimator_class": "Thing",
        "estimator_package": "notinstalled",
        "params": {},
    }
    with pytest.raises(UnsupportedEstimatorError, match="notinstalled.Thing"):
        SklearnSerializer().deserialize(data)
    assert "notinstalled" not in sys.modules


def test_unknown_root_class_raises_unsupported_estimator():
    with pytest.raises(UnsupportedEstimatorError, match="Nope"):
        SklearnSerializer().deserialize({"estimator_class": "Nope", "params": {}})


@pytest.mark.parametrize("with_package", [True, False])
def test_unknown_nested_class_raises_instead_of_loading_a_dict(with_package):
    serializer = SklearnSerializer()
    serialized = _json(
        serializer.serialize(
            Pipeline([("scale", StandardScaler()), ("sk", MinMaxScaler())]).fit(X)
        )
    )
    step = serialized["params"]["steps"][0][1]
    step["estimator_class"] = "Nope"
    serialized["param_types"]["steps"][0][1] = "Nope"
    if not with_package:
        step.pop("estimator_package")
    with pytest.raises(UnsupportedEstimatorError, match="Nope"):
        serializer.deserialize(serialized)


# ==== Function allowlist ====


def _select_k_best_dict(score_func_ref=None):
    serialized = SklearnSerializer().serialize(SelectKBest(chi2, k=2).fit(X, y))
    if score_func_ref is not None:
        serialized["params"]["score_func"] = score_func_ref
    return serialized


def test_allowlisted_function_roundtrips():
    model = SelectKBest(chi2, k=2).fit(X, y)
    back = _roundtrip(SklearnSerializer(), model)
    assert back.score_func is chi2
    np.testing.assert_array_equal(back.transform(X), model.transform(X))


@pytest.mark.parametrize(
    "ref",
    [
        {"module": "os", "name": "getcwd"},
        {"module": "numpy", "name": "_private"},
        {"module": "numpy", "name": "ndarray"},  # a class, not a function
        {"module": "numpy.f2py.__main__", "name": "main"},
        {"module": "notinstalled_mod", "name": "f"},
        {"module": "sklearn", "name": None},
    ],
)
def test_disallowed_function_is_refused(ref):
    manager = SerializationManager(SklearnSerializer())
    with pytest.raises(DeserializationError, match="not allowed"):
        manager.deserialize(json.dumps(_select_k_best_dict(ref)))
    assert "notinstalled_mod" not in sys.modules


def test_trusted_function_module(tmp_path, monkeypatch):
    (tmp_path / "om_trusted_mod.py").write_text("def double(X):\n    return X * 2\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    import om_trusted_mod  # type: ignore[import-not-found]

    try:
        model = FunctionTransformer(func=om_trusted_mod.double).fit(X)
        serialized = _json(SklearnSerializer().serialize(model))

        # Refused by default, even though the module is already imported.
        with pytest.raises(DeserializationError, match="om_trusted_mod.double"):
            SklearnSerializer().deserialize(copy.deepcopy(serialized))

        # Trusted: imported on demand.
        del sys.modules["om_trusted_mod"]
        trusted = SklearnSerializer(trusted_function_modules=["om_trusted_mod"])
        back = trusted.deserialize(serialized)
        np.testing.assert_array_equal(back.transform(X), X * 2)
    finally:
        sys.modules.pop("om_trusted_mod", None)


# ==== Kernels ====


def test_kernel_type_must_be_a_kernel_class():
    serializer = SklearnSerializer()
    model = GaussianProcessRegressor(kernel=RBF() + WhiteKernel()).fit(X, y)
    serialized = serializer.serialize(model)
    serialized["params"]["kernel"]["params"]["k1"]["kernel_type"] = "np"
    with pytest.raises(DeserializationError, match="np"):
        serializer.deserialize(serialized)
