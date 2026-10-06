"""
KernelDensity's search tree (`tree_`) must round-trip with its metric, leaf_size and sample
weights; the weights are stored only inside the tree.
"""

import json

import numpy as np
import pytest
from sklearn.neighbors import BallTree, KDTree, KernelDensity

from openmodels import SerializationManager, SklearnSerializer

rng = np.random.RandomState(0)
X = rng.rand(40, 3)
WEIGHTS = rng.rand(40)

CASES = {
    "default": (lambda: KernelDensity(), None),
    "ball_tree": (lambda: KernelDensity(algorithm="ball_tree"), None),
    "kd_tree_manhattan": (
        lambda: KernelDensity(algorithm="kd_tree", metric="manhattan"),
        None,
    ),
    "leaf_size": (lambda: KernelDensity(leaf_size=5), None),
    "sample_weight": (lambda: KernelDensity(), WEIGHTS),
    "bandwidth_scott": (lambda: KernelDensity(bandwidth="scott"), None),
}


def _fit(case):
    make, weights = CASES[case]
    return make().fit(X, sample_weight=weights)


def _roundtrip(model, fmt="json"):
    manager = SerializationManager(SklearnSerializer())
    return manager.deserialize(
        manager.serialize(model, format_name=fmt), format_name=fmt
    )


def _assert_same(loaded, original):
    np.testing.assert_allclose(loaded.score_samples(X), original.score_samples(X))
    assert type(loaded.tree_) is type(original.tree_)
    for a, b in zip(original.tree_.get_arrays(), loaded.tree_.get_arrays()):
        np.testing.assert_array_equal(b, a)


@pytest.mark.parametrize("case", CASES)
def test_kernel_density_roundtrips(case):
    model = _fit(case)
    _assert_same(_roundtrip(model), model)


def test_kernel_density_roundtrips_with_pickle():
    model = _fit("sample_weight")
    _assert_same(_roundtrip(model, "pickle"), model)


def test_tree_payload_holds_sample_weights():
    serialized = SklearnSerializer().serialize(_fit("sample_weight"))
    assert serialized["attribute_types"]["tree_"] == "KDTree"
    np.testing.assert_allclose(
        serialized["attributes"]["tree_"]["sample_weight"], WEIGHTS
    )

    unweighted = SklearnSerializer().serialize(_fit("ball_tree"))
    assert unweighted["attribute_types"]["tree_"] == "BallTree"
    assert unweighted["attributes"]["tree_"]["sample_weight"] is None


def test_file_written_by_earlier_versions_loads_unweighted_model():
    # 0.2.x wrote KDTree payloads without "sample_weight"; such files never loaded before.
    model = _fit("kd_tree_manhattan")
    serializer = SklearnSerializer()
    serialized = json.loads(json.dumps(serializer.serialize(model)))
    del serialized["attributes"]["tree_"]["sample_weight"]
    _assert_same(serializer.deserialize(serialized), model)


def test_neighbors_tree_payload_unchanged_for_old_readers():
    # Neighbors estimators still write a KDTree with "data"/"data_dtype" (all 0.2.2 reads).
    from sklearn.neighbors import KNeighborsClassifier

    model = KNeighborsClassifier(algorithm="kd_tree").fit(
        X, (X[:, 0] > 0.5).astype(int)
    )
    payload = SklearnSerializer().serialize(model)["attributes"]["_tree"]
    assert {"data", "data_dtype"} <= set(payload)
    assert isinstance(KDTree(np.array(payload["data"])), KDTree)
    assert not isinstance(model._tree, BallTree)
