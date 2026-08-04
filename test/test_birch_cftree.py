"""
Structural invariant checks for Birch's root_/dummy_leaf_ CF-tree round-trip.

Generic predict()/transform() checks don't exercise root_/dummy_leaf_ at all (those methods
only use subcluster_centers_/subcluster_labels_/_subcluster_norms), so a broken leaf chain
would pass predict()-only testing silently. These checks walk the tree directly.
"""

import numpy as np
from sklearn.cluster import Birch

from openmodels import SerializationManager, SklearnSerializer


def _leaves(model):
    leaves = []
    leaf = model.dummy_leaf_.next_leaf_
    while leaf is not None:
        leaves.append(leaf)
        leaf = leaf.next_leaf_
    return leaves


def test_birch_leaf_chain_round_trips():
    X = np.random.RandomState(0).rand(200, 4)
    model = Birch(n_clusters=3, threshold=0.3, branching_factor=10)
    model.fit(X)

    leaves_before = _leaves(model)
    n_subclusters_before = sum(len(leaf.subclusters_) for leaf in leaves_before)

    manager = SerializationManager(SklearnSerializer())
    restored = manager.deserialize(manager.serialize(model))

    leaves_after = _leaves(restored)
    n_subclusters_after = sum(len(leaf.subclusters_) for leaf in leaves_after)

    assert len(leaves_after) == len(leaves_before)
    assert n_subclusters_after == n_subclusters_before
    assert restored.dummy_leaf_.next_leaf_ is leaves_after[0]
    assert leaves_after[0].prev_leaf_ is restored.dummy_leaf_
    assert leaves_after[-1].next_leaf_ is None

    np.testing.assert_allclose(model.subcluster_centers_, restored.subcluster_centers_)


def test_birch_partial_fit_after_round_trip():
    # This is the scenario the original bug affected: predict()/transform() never touch
    # root_/dummy_leaf_, but partial_fit() walks the leaf chain via Birch._get_leaves() to
    # recompute subcluster_centers_, which crashed with "need at least one array to
    # concatenate" when the tree structure was lost on deserialize.
    X = np.random.RandomState(1).rand(100, 3)
    model = Birch(n_clusters=2, threshold=0.5, branching_factor=10)
    model.fit(X)

    manager = SerializationManager(SklearnSerializer())
    restored = manager.deserialize(manager.serialize(model))

    restored.partial_fit(X[:10])
    assert restored.subcluster_centers_.shape[1] == X.shape[1]


def test_birch_float32_center_dtype_preserved():
    X = np.random.RandomState(2).rand(60, 4).astype(np.float32)
    model = Birch(n_clusters=2, threshold=0.3, branching_factor=10)
    model.fit(X)

    manager = SerializationManager(SklearnSerializer())
    restored = manager.deserialize(manager.serialize(model))

    for leaf in _leaves(restored):
        assert leaf.centroids_.dtype == np.float32
        for sub in leaf.subclusters_:
            assert sub.centroid_.dtype == np.float32
