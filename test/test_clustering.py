import pytest
import random
from sklearn.utils.discovery import all_estimators
from sklearn.datasets import make_blobs
from sklearn.feature_extraction import FeatureHasher
from openmodels.test_helpers import run_test_model
from openmodels.serializers.sklearn.sklearn_serializer import NOT_SUPPORTED_ESTIMATORS

# Get all cluster estimators, filtering out not supported clusters
CLUSTERS = [
    cls for name, cls in all_estimators(type_filter="cluster")
    if name not in NOT_SUPPORTED_ESTIMATORS
]

@pytest.fixture(scope="module")
def data():
    # Generate test data
    x, _ = make_blobs(
        n_samples=30,
        n_features=4,
        centers=3,
        random_state=42,
    )

    feature_hasher = FeatureHasher(n_features=4)
    features = []
    for _ in range(0, 100):
        features.append(
            {
                "a": random.randint(0, 2),
                "b": random.randint(3, 5),
                "c": random.randint(6, 8),
            }
        )
    x_sparse = feature_hasher.transform(iter(features))

    return x, x_sparse


@pytest.mark.parametrize("Cluster", CLUSTERS)
def test_clusterer_fit_predict(Cluster, data):
    x, x_sparse = data
    cluster = Cluster()

    # Run the test model
    run_test_model(cluster, x, None, x_sparse, None, f"{Cluster.__name__.lower()}.json")
