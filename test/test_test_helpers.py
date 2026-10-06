"""
Tests for openmodels.test_helpers.run_test_model itself: it must round-trip the dense fit and
the sparse fit separately, never the caller's own instance.
"""

import os

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils.validation import check_is_fitted

from openmodels import test_helpers
from openmodels.serializers.sklearn.sklearn_serializer import SklearnSerializer

rng = np.random.RandomState(0)
X = rng.rand(60, 3)
Y = X @ np.array([1.0, -2.0, 0.5])
X_SPARSE = csr_matrix(rng.rand(60, 3) * (rng.rand(60, 3) > 0.5))
Y_SPARSE = rng.rand(60)


def test_dense_and_sparse_fits_are_both_round_tripped(monkeypatch):
    fit_methods = []
    original_serialize = SklearnSerializer.serialize

    def recording_serialize(self, model):
        serialized = original_serialize(self, model)
        fit_methods.append(serialized["attributes"]["_fit_method"])
        return serialized

    monkeypatch.setattr(SklearnSerializer, "serialize", recording_serialize)
    test_helpers.run_test_model(
        KNeighborsRegressor(), X, Y, X_SPARSE, Y_SPARSE, "helper_knn"
    )

    # Dense data gives a search tree, sparse forces brute force: both fits must be saved.
    assert fit_methods == ["kd_tree", "brute"]


def test_sparse_fit_skipped_without_y_sparse(monkeypatch):
    fit_methods = []
    original_serialize = SklearnSerializer.serialize

    def recording_serialize(self, model):
        serialized = original_serialize(self, model)
        fit_methods.append(serialized["attributes"]["_fit_method"])
        return serialized

    monkeypatch.setattr(SklearnSerializer, "serialize", recording_serialize)
    test_helpers.run_test_model(
        KNeighborsRegressor(), X, Y, X_SPARSE, None, "helper_knn"
    )
    assert fit_methods == ["kd_tree"]


def test_callers_model_is_left_unfitted():
    model = KNeighborsRegressor()
    test_helpers.run_test_model(model, X, Y, X_SPARSE, Y_SPARSE, "helper_knn")
    with pytest.raises(NotFittedError):
        check_is_fitted(model)


def test_temp_file_removed_and_failure_labelled(monkeypatch):
    def failing_compare(*args, **kwargs):
        raise AssertionError("mismatch")

    monkeypatch.setattr(test_helpers, "run_test_predictions", failing_compare)
    with pytest.raises(AssertionError, match=r"\[helper_fail\] mismatch"):
        test_helpers.run_test_model(
            KNeighborsRegressor(), X, Y, None, None, "helper_fail"
        )
    assert not os.path.exists("./test/temp/helper_fail.json")
