import pytest
from sklearn.cross_decomposition import CCA, PLSCanonical, PLSRegression, PLSSVD

from openmodels.test_helpers import roundtrip_fit


@pytest.fixture(scope="module", autouse=True)
def _roundtrip_pls_family():
    with roundtrip_fit(PLSRegression, PLSCanonical, CCA, PLSSVD):
        yield
