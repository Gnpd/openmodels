"""
Gaussian-process kernels must round-trip wherever they're used (GPR, GPC, KernelRidge), for
every kernel class, at top level and nested, including array hyperparameters and lists of
kernels. A file can only make loading build a concrete scikit-learn Kernel class.
"""

import json

import numpy as np
import pytest
from sklearn.gaussian_process import (
    GaussianProcessClassifier,
    GaussianProcessRegressor,
)
from sklearn.gaussian_process import kernels as K
from sklearn.kernel_ridge import KernelRidge

from openmodels import SerializationManager, SklearnSerializer
from openmodels.exceptions import DeserializationError

rng = np.random.RandomState(0)
X = rng.rand(30, 3)
y_reg = X @ np.array([1.0, -2.0, 0.5]) + 0.1 * rng.rand(30)
y_clf = (X[:, 0] > 0.5).astype(int)


def _roundtrip(model, fmt="json"):
    manager = SerializationManager(SklearnSerializer())
    return manager.deserialize(
        manager.serialize(model, format_name=fmt), format_name=fmt
    )


GPR_KERNELS = {
    "RBF": lambda: K.RBF(),
    "RBF_anisotropic": lambda: K.RBF(length_scale=[1.0, 1.0, 1.0]),
    "Matern": lambda: K.Matern(nu=1.5),
    "RationalQuadratic": lambda: K.RationalQuadratic(),
    "Exponentiation": lambda: K.Exponentiation(K.RBF(), 2),
    "PairwiseKernel": lambda: K.PairwiseKernel(metric="laplacian"),
    "Constant_x_RBF_plus_White": lambda: K.ConstantKernel() * K.RBF() + K.WhiteKernel(),
    "Matern_plus_White": lambda: K.Matern() + K.WhiteKernel(),
}


@pytest.mark.parametrize("kernel", GPR_KERNELS)
def test_gpr_kernel_roundtrips(kernel):
    model = GaussianProcessRegressor(kernel=GPR_KERNELS[kernel](), random_state=0)
    model.fit(X, y_reg)
    loaded = _roundtrip(model)
    assert type(loaded.kernel_) is type(model.kernel_)
    assert type(loaded.kernel) is type(model.kernel)
    np.testing.assert_allclose(loaded.predict(X), model.predict(X))
    np.testing.assert_allclose(loaded.kernel_.theta, model.kernel_.theta)


def test_gpr_kernel_roundtrips_with_pickle():
    model = GaussianProcessRegressor(kernel=K.Matern(), random_state=0).fit(X, y_reg)
    loaded = _roundtrip(model, "pickle")
    np.testing.assert_allclose(loaded.predict(X), model.predict(X))


def test_gpc_matern_roundtrips():
    model = GaussianProcessClassifier(kernel=K.Matern(), random_state=0).fit(X, y_clf)
    loaded = _roundtrip(model)
    np.testing.assert_allclose(loaded.predict_proba(X), model.predict_proba(X))


@pytest.mark.parametrize(
    "kernel",
    [K.Matern(), K.ExpSineSquared(), K.RationalQuadratic()],
    ids=lambda k: type(k).__name__,
)
def test_kernel_ridge_kernel_roundtrips(kernel):
    model = KernelRidge(kernel=kernel).fit(X, y_reg)
    loaded = _roundtrip(model)
    assert type(loaded.kernel) is type(kernel)
    np.testing.assert_allclose(loaded.predict(X), model.predict(X))


def test_compound_kernel_roundtrips():
    # CompoundKernel's `kernels` param is a list of kernels.
    kernel = K.CompoundKernel([K.RBF(), K.WhiteKernel()])
    loaded = _roundtrip(KernelRidge(kernel=kernel)).kernel
    assert type(loaded) is K.CompoundKernel
    assert [type(k) for k in loaded.kernels] == [K.RBF, K.WhiteKernel]
    np.testing.assert_allclose(loaded(X), kernel(X))


@pytest.mark.parametrize("kernel_type", ["np", "Kernel", "_private", "MyKernel", None])
def test_non_kernel_types_refused(kernel_type):
    model = GaussianProcessRegressor(kernel=K.RBF()).fit(X, y_reg)
    serialized = json.loads(json.dumps(SklearnSerializer().serialize(model)))
    serialized["attributes"]["kernel_"]["kernel_type"] = kernel_type
    with pytest.raises(DeserializationError, match="Unknown kernel type"):
        SklearnSerializer().deserialize(serialized)


def test_top_level_matern_written_by_earlier_versions_loads():
    # 0.2.x wrote a top-level Matern exactly in this shape (and couldn't load it back).
    model = GaussianProcessRegressor(kernel=K.Matern(), random_state=0).fit(X, y_reg)
    serialized = json.loads(json.dumps(SklearnSerializer().serialize(model)))
    assert serialized["attribute_types"]["kernel_"] == "Matern"
    assert serialized["attributes"]["kernel_"]["kernel_type"] == "Matern"
    loaded = SklearnSerializer().deserialize(serialized)
    np.testing.assert_allclose(loaded.predict(X), model.predict(X))
