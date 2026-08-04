import pytest
from sklearn.cluster import Birch

from openmodels.test_helpers import roundtrip_fit


@pytest.fixture(scope="module", autouse=True)
def _roundtrip_birch():
    with roundtrip_fit(Birch):
        yield
