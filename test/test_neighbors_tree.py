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
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
    KNeighborsTransformer,
    LocalOutlierFactor,
    NearestNeighbors,
    RadiusNeighborsClassifier,
    RadiusNeighborsRegressor,
    RadiusNeighborsTransformer,
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
    "radius_transformer_kd_manhattan": (
        lambda: RadiusNeighborsTransformer(
            radius=1.0, algorithm="kd_tree", metric="manhattan"
        ),
        None,
    ),
    # BallTree cases: the tree isn't written, only rebuilt on load.
    "knn_classifier_ball_tree": (
        lambda: KNeighborsClassifier(algorithm="ball_tree"),
        y,
    ),
    "radius_classifier_ball_tree": (
        lambda: RadiusNeighborsClassifier(radius=1.0, algorithm="ball_tree"),
        y,
    ),
    "nearest_neighbors_ball_tree": (
        lambda: NearestNeighbors(algorithm="ball_tree"),
        None,
    ),
    "local_outlier_factor_ball_tree": (
        lambda: LocalOutlierFactor(novelty=True, algorithm="ball_tree"),
        None,
    ),
    "kneighbors_transformer_ball_tree": (
        lambda: KNeighborsTransformer(algorithm="ball_tree"),
        None,
    ),
    "radius_transformer_ball_tree": (
        lambda: RadiusNeighborsTransformer(radius=1.0, algorithm="ball_tree"),
        None,
    ),
}

BALL_TREE_CASES = [case for case in CASES if case.endswith("ball_tree")]


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


@pytest.mark.parametrize("case", BALL_TREE_CASES)
def test_ball_tree_not_written(case):
    model = _fit(case)
    assert model._fit_method == "ball_tree"
    serialized = SklearnSerializer().serialize(model)
    assert "_tree" not in serialized["attributes"]


def test_auto_haversine_uses_ball_tree_and_roundtrips():
    coords = rng.rand(40, 2)  # (lat, lon) in radians
    labels = (coords[:, 0] > 0.5).astype(int)
    model = KNeighborsClassifier(metric="haversine").fit(coords, labels)
    assert model._fit_method == "ball_tree"
    loaded = _roundtrip(model)
    np.testing.assert_array_equal(loaded.predict(coords), model.predict(coords))
    d1, i1 = model.kneighbors(coords)
    d2, i2 = loaded.kneighbors(coords)
    np.testing.assert_allclose(d2, d1)
    np.testing.assert_array_equal(i2, i1)


@pytest.mark.parametrize(
    "model",
    [
        Isomap(n_neighbors=5, neighbors_algorithm="ball_tree"),
        LocallyLinearEmbedding(n_neighbors=8, neighbors_algorithm="ball_tree"),
    ],
    ids=["Isomap", "LocallyLinearEmbedding"],
)
def test_nested_ball_tree_roundtrips(model):
    model.fit(X)
    loaded = _roundtrip(model)
    np.testing.assert_allclose(loaded.transform(X), model.transform(X))


def test_radius_neighbors_transformer_brute_roundtrips():
    # Its training data (_fit_X) wasn't saved before, so a brute-force model couldn't transform.
    model = RadiusNeighborsTransformer(radius=1.0).fit(csr_matrix(X))
    assert model._fit_method == "brute"
    loaded = _roundtrip(model)
    np.testing.assert_allclose(
        loaded.transform(csr_matrix(X)).toarray(),
        model.transform(csr_matrix(X)).toarray(),
    )
