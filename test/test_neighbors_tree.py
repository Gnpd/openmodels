"""
Neighbors estimators' search trees (`_tree`) are rebuilt on load from the estimator's own
state (`_fit_X`, `leaf_size`, `effective_metric_`, `effective_metric_params_`), exactly as
scikit-learn builds them. All fits here are on dense data, where `algorithm="auto"` picks a
tree: sparse input forces brute force and would hide a missing or wrong tree.
"""

import json

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
    KNeighborsTransformer,
    LocalOutlierFactor,
    NearestNeighbors,
    RadiusNeighborsClassifier,
    RadiusNeighborsRegressor,
)

from openmodels import SerializationManager, SklearnSerializer

rng = np.random.RandomState(0)
X = rng.rand(60, 4)
y = (X[:, 0] + X[:, 1] > 1).astype(int)
yr = X @ np.array([1.0, -2.0, 0.5, 3.0])

CASES = {
    "knn_regressor_auto": (lambda: KNeighborsRegressor(), yr),
    "knn_regressor_kd_manhattan": (
        lambda: KNeighborsRegressor(algorithm="kd_tree", metric="manhattan"),
        yr,
    ),
    "knn_regressor_ball_tree": (lambda: KNeighborsRegressor(algorithm="ball_tree"), yr),
    "radius_regressor_kd": (
        lambda: RadiusNeighborsRegressor(radius=1.0, algorithm="kd_tree"),
        yr,
    ),
    "knn_classifier_kd_manhattan": (
        lambda: KNeighborsClassifier(algorithm="kd_tree", metric="manhattan"),
        y,
    ),
    "knn_classifier_leaf_size": (
        lambda: KNeighborsClassifier(algorithm="kd_tree", leaf_size=5),
        y,
    ),
    "radius_classifier_kd_chebyshev": (
        lambda: RadiusNeighborsClassifier(
            radius=1.0, algorithm="kd_tree", metric="chebyshev"
        ),
        y,
    ),
    "nearest_neighbors_kd_p3": (
        lambda: NearestNeighbors(algorithm="kd_tree", p=3),
        None,
    ),
    "local_outlier_factor_auto": (lambda: LocalOutlierFactor(novelty=True), None),
    "kneighbors_transformer_kd": (
        lambda: KNeighborsTransformer(algorithm="kd_tree"),
        None,
    ),
}


def _fit(case):
    make, target = CASES[case]
    model = make()
    return model.fit(X) if target is None else model.fit(X, target)


def _roundtrip(model, fmt="json"):
    manager = SerializationManager(SklearnSerializer())
    return manager.deserialize(
        manager.serialize(model, format_name=fmt), format_name=fmt
    )


def _assert_same_neighbors(loaded, original):
    if hasattr(original, "kneighbors"):
        d1, i1 = original.kneighbors(X[:20])
        d2, i2 = loaded.kneighbors(X[:20])
        np.testing.assert_allclose(d2, d1)
        np.testing.assert_array_equal(i2, i1)
    else:
        d1, i1 = original.radius_neighbors(X[:20])
        d2, i2 = loaded.radius_neighbors(X[:20])
        for a, b in zip(d1, d2):
            np.testing.assert_allclose(b, a)
        for a, b in zip(i1, i2):
            np.testing.assert_array_equal(b, a)

    if isinstance(original, LocalOutlierFactor):
        np.testing.assert_allclose(loaded.score_samples(X), original.score_samples(X))
    elif hasattr(original, "predict"):
        np.testing.assert_allclose(loaded.predict(X), original.predict(X))


def _assert_same_tree(loaded, original):
    assert type(loaded._tree) is type(original._tree)
    for a, b in zip(original._tree.get_arrays(), loaded._tree.get_arrays()):
        np.testing.assert_array_equal(b, a)


@pytest.mark.parametrize("case", CASES)
def test_neighbors_tree_rebuilt_identically(case):
    model = _fit(case)
    assert model._fit_method in ("kd_tree", "ball_tree")  # the case must use a tree
    loaded = _roundtrip(model)
    _assert_same_neighbors(loaded, model)
    _assert_same_tree(loaded, model)


def test_neighbors_tree_rebuilt_with_pickle():
    model = _fit("knn_regressor_kd_manhattan")
    loaded = _roundtrip(model, "pickle")
    _assert_same_neighbors(loaded, model)
    _assert_same_tree(loaded, model)


def test_file_with_stored_tree_gets_correct_metric():
    # Classifiers keep writing `_tree` (so 0.2.2 readers still load these files). That stored
    # tree only holds the data - 0.2.x rebuilt it as KDTree(data), i.e. euclidean - so it must
    # be ignored in favour of a rebuild with the real metric.
    model = _fit("knn_classifier_kd_manhattan")
    serializer = SklearnSerializer()
    serialized = json.loads(json.dumps(serializer.serialize(model)))
    assert serialized["attribute_types"]["_tree"] == "KDTree"

    loaded = serializer.deserialize(serialized)
    _assert_same_neighbors(loaded, model)
    _assert_same_tree(loaded, model)


def test_brute_model_has_no_tree():
    model = KNeighborsClassifier().fit(csr_matrix(X), y)
    assert model._fit_method == "brute"
    loaded = _roundtrip(model)
    assert loaded._tree is None
    np.testing.assert_array_equal(loaded.predict(X), model.predict(X))


def test_file_without_rebuild_inputs_still_loads():
    model = _fit("knn_classifier_leaf_size")
    serializer = SklearnSerializer()
    serialized = json.loads(json.dumps(serializer.serialize(model)))
    del serialized["attributes"]["effective_metric_"]

    loaded = serializer.deserialize(serialized)
    # No rebuild without the metric: the stored tree is used, as before.
    assert loaded._tree is not None
    np.testing.assert_array_equal(loaded.predict(X), model.predict(X))


@pytest.mark.xfail(
    strict=True,
    reason="BUG_AUDIT.md #7: LOF still writes _tree, and a BallTree can't be serialized",
)
def test_local_outlier_factor_ball_tree():
    model = LocalOutlierFactor(novelty=True, algorithm="ball_tree").fit(X)
    loaded = _roundtrip(model)
    _assert_same_neighbors(loaded, model)
