"""
Tests for openmodels.test_helpers itself: run_test_model must round-trip the dense fit and the
sparse fit separately, never the caller's own instance; roundtrip_fit must leave an instance
with exactly the state a loaded model has.
"""

import inspect
import os

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
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


# ==== roundtrip_fit ====


def test_roundtrip_fit_drops_fitted_state_that_isnt_saved():
    """An attribute fit creates but openmodels doesn't save (KMeans' training-only _tol) is
    missing afterwards, as on a loaded model. Saved state is the loaded copy's."""
    plain = KMeans(n_clusters=2, n_init=1, random_state=0).fit(X)
    assert hasattr(plain, "_tol")

    model = KMeans(n_clusters=2, n_init=1, random_state=0)
    with test_helpers.roundtrip_fit(KMeans):
        model.fit(X)
    assert not hasattr(model, "_tol")
    np.testing.assert_array_equal(model.cluster_centers_, plain.cluster_centers_)
    np.testing.assert_array_equal(model.predict(X), plain.predict(X))


def test_roundtrip_fit_keeps_params_and_untouched_configuration():
    hidden = (3,)
    model = MLPRegressor(hidden_layer_sizes=hidden, max_iter=5, random_state=0)
    with test_helpers.roundtrip_fit(MLPRegressor):
        model.fit(X, Y)
    assert model.hidden_layer_sizes is hidden  # JSON would have made it a list

    scaler = StandardScaler().set_output(transform="pandas")
    with test_helpers.roundtrip_fit(StandardScaler):
        scaler.fit(X)
    assert isinstance(scaler.transform(X), pd.DataFrame)


def test_roundtrip_fit_restores_methods_and_keeps_signature():
    original_fit = KMeans.__dict__["fit"]
    with test_helpers.roundtrip_fit(KMeans):
        # scikit-learn's check_fit_score_takes_y inspects fit's signature
        params = list(inspect.signature(KMeans(n_init=1).fit).parameters)
        assert params[:2] == ["X", "y"]
    assert KMeans.__dict__["fit"] is original_fit
