import numpy as np
import pytest
import scipy.sparse as sp

from openmodels import SerializationManager, SklearnSerializer
from sklearn.neighbors import KNeighborsClassifier


@pytest.mark.parametrize(
    "sparse_cls", [sp.csr_matrix, sp.csc_matrix, sp.csr_array, sp.csc_array]
)
def test_estimator_with_sparse_fitted_attribute_round_trips(sparse_cls):
    # KNeighborsClassifier stores its training data (whatever sparse container it was fit
    # with) as the fitted attribute _fit_X - this exercises openmodels's sparse serializer
    # directly, independent of which format the caller happened to pass in.
    X = np.random.RandomState(0).rand(30, 4)
    X[X < 0.5] = 0
    X_sparse = sparse_cls(X)
    y = np.arange(30) % 3

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_sparse, y)

    manager = SerializationManager(SklearnSerializer())
    restored = manager.deserialize(manager.serialize(model))

    np.testing.assert_array_equal(
        model.predict(X_sparse[:5]), restored.predict(X_sparse[:5])
    )
