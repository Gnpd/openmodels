"""
SciPy frozen distributions (as used in RandomizedSearchCV's param_distributions) must
round-trip, and a model file must never be able to make loading call any other `scipy.stats`
callable.
"""

import json

import numpy as np
import pytest
import scipy.stats as st
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import FunctionTransformer

from openmodels import SerializationManager, SklearnSerializer
from openmodels.exceptions import DeserializationError

X = np.random.RandomState(0).rand(40, 3)
y = (X[:, 0] > 0.5).astype(int)


def _roundtrip(model):
    manager = SerializationManager(SklearnSerializer())
    return manager.deserialize(manager.serialize(model))


def _search_dict(param_distributions):
    """A serialized (JSON round-tripped) unfitted RandomizedSearchCV."""
    model = RandomizedSearchCV(LogisticRegression(), param_distributions, n_iter=2)
    return json.loads(json.dumps(SklearnSerializer().serialize(model)))


# ==== security ====


@pytest.mark.parametrize(
    "name", ["describe", "zscore", "_private", "nonexistent", None]
)
@pytest.mark.parametrize("tag", ["rv_continuous_frozen", "rv_discrete_frozen"])
def test_non_distribution_names_refused(name, tag):
    crafted = {"dist_name": name, "args": [[1, 2, 3]], "kwargs": {}}
    with pytest.raises(DeserializationError, match="Unknown distribution"):
        SklearnSerializer().convert_from_serializable(crafted, tag)


def test_non_distribution_name_inside_dict_param_refused():
    serialized = _search_dict({"C": st.uniform(0.1, 1)})
    serialized["params"]["param_distributions"]["C"]["dist_name"] = "describe"
    with pytest.raises(DeserializationError, match="Unknown distribution 'describe'"):
        SklearnSerializer().deserialize(serialized)


def test_legacy_scipy_dist_tag_calls_nothing(monkeypatch):
    calls = []
    monkeypatch.setattr(st, "describe", lambda *a, **k: calls.append(a))
    crafted = {"dist_name": "describe", "args": [[1, 2, 3]], "kwargs": {}}
    assert (
        SklearnSerializer().convert_from_serializable(crafted, "scipy_dist") == crafted
    )
    assert calls == []


# ==== round trips ====


@pytest.mark.parametrize(
    "dist",
    [
        st.uniform(0.1, 1),
        st.loguniform(1e-3, 1e2),
        st.norm(loc=0, scale=2),
        st.randint(1, 10),
        st.poisson(3),
    ],
    ids=lambda d: d.dist.name,
)
def test_distribution_roundtrips(dist):
    serializer = SklearnSerializer()
    written = json.loads(json.dumps(serializer.convert_to_serializable(dist)))
    loaded = serializer.convert_from_serializable(
        written, serializer._get_nested_types(dist)
    )
    assert type(loaded) is type(dist)
    assert loaded.dist.name == dist.dist.name
    assert list(loaded.args) == list(dist.args) and loaded.kwds == dist.kwds
    np.testing.assert_array_equal(
        loaded.rvs(size=5, random_state=0), dist.rvs(size=5, random_state=0)
    )


@pytest.mark.parametrize("fitted", [False, True], ids=["unfitted", "fitted"])
@pytest.mark.parametrize(
    "param_distributions",
    [
        {"C": st.loguniform(1e-2, 1e1), "max_iter": st.randint(50, 200)},
        [{"C": st.loguniform(1e-2, 1e1)}, {"max_iter": st.randint(50, 200)}],
    ],
    ids=["dict", "list_of_dicts"],
)
def test_randomized_search_roundtrips(param_distributions, fitted):
    model = RandomizedSearchCV(
        LogisticRegression(), param_distributions, n_iter=3, cv=2, random_state=0
    )
    if fitted:
        model.fit(X, y)
    loaded = _roundtrip(model)

    grids = loaded.param_distributions
    for grid in grids if isinstance(grids, list) else [grids]:
        assert all(hasattr(v, "rvs") for v in grid.values())
    clone(loaded)
    if fitted:
        np.testing.assert_array_equal(loaded.predict(X), model.predict(X))

    # Refitting the loaded search samples the same candidates as the original.
    refit_original = clone(model).fit(X, y)
    refit_loaded = clone(loaded).fit(X, y)
    assert refit_loaded.cv_results_["params"] == refit_original.cv_results_["params"]


def test_param_distributions_written_by_earlier_versions_load():
    # 0.2.x wrote continuous distributions in exactly this shape inside the dict param.
    serialized = _search_dict({"C": st.uniform(0.1, 1)})
    serialized["params"]["param_distributions"]["C"] = {
        "dist_name": "uniform",
        "args": [0.1, 1],
        "kwargs": {},
    }
    loaded = SklearnSerializer().deserialize(serialized)
    assert loaded.param_distributions["C"].dist.name == "uniform"


def test_dict_with_other_keys_left_unchanged():
    kw_args = {"dist_name": "x", "args": [1]}  # not the exact three-key shape
    loaded = _roundtrip(FunctionTransformer(kw_args=kw_args))
    assert loaded.kw_args == kw_args
