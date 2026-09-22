import json
import platform
import warnings
from datetime import datetime

import numpy as np
import pytest
import scipy
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from openmodels.core import SerializationManager
from openmodels.serializers.sklearn.sklearn_serializer import (
    SklearnSerializer,
    _openmodels_version,
)


def get_fitted_model():
    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    y = np.array([0, 1, 1, 0])
    model = LogisticRegression()
    model.fit(X, y)
    return model, X


def get_fitted_pipeline():
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    y = np.array([0, 1, 1, 0])
    pipeline = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    pipeline.fit(X, y)
    return pipeline, X


def _count_key(obj, key):
    """Recursively count how many dicts anywhere in obj contain `key`."""
    count = 0
    if isinstance(obj, dict):
        if key in obj:
            count += 1
        for v in obj.values():
            count += _count_key(v, key)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            count += _count_key(item, key)
    return count


def test_metadata_present_once_on_nested_estimator():
    pipeline, _ = get_fitted_pipeline()
    serialized = SklearnSerializer().serialize(pipeline)

    assert "metadata" in serialized
    assert _count_key(serialized, "metadata") == 1


def test_serialize_accepts_optional_metadata():
    model, _ = get_fitted_model()
    manager = SerializationManager(SklearnSerializer())

    result = json.loads(
        manager.serialize(
            model,
            format_name="json",
            metadata={
                "title": "My model",
                "description": "A test model",
                "author": {"name": "Jane Doe", "email": "jane@example.com"},
                "domain": "should-not-win",
            },
        )
    )

    metadata = result["metadata"]
    assert metadata["title"] == "My model"
    assert metadata["description"] == "A test model"
    assert metadata["author"] == {"name": "Jane Doe", "email": "jane@example.com"}
    # Autofilled fields always win on collision with user-supplied metadata.
    assert metadata["domain"] == "sklearn"


def test_save_accepts_optional_metadata(tmp_path):
    model, X = get_fitted_model()
    manager = SerializationManager(SklearnSerializer())
    file_path = tmp_path / "model.json"

    manager.save(
        model, file_path, format_name="json", metadata={"title": "Saved model"}
    )

    result = json.loads(file_path.read_text())
    assert result["metadata"]["title"] == "Saved model"

    loaded_model = manager.load(file_path, format_name="json")
    assert np.array_equal(model.predict(X), loaded_model.predict(X))


def test_deserialize_old_flat_format_still_works():
    model, X = get_fitted_model()
    serialized = SklearnSerializer().serialize(model)

    # Reconstruct the pre-v2 shape: bookkeeping fields flat at the top level, no "metadata" key.
    old_flat = {k: v for k, v in serialized.items() if k != "metadata"}
    old_flat.update(serialized["metadata"])
    old_flat["openmodels_format_version"] = 1
    # Pre-v3 files have no domain_version; producer_version held the scikit-learn version.
    old_flat.pop("domain_version")
    old_flat["producer_version"] = sklearn.__version__

    deserialized_model = SklearnSerializer().deserialize(old_flat)
    assert np.array_equal(model.predict(X), deserialized_model.predict(X))

    old_flat["producer_version"] = "0.0.1"
    with pytest.warns(UserWarning, match="Version mismatch"):
        SklearnSerializer().deserialize(old_flat)


def test_metadata_includes_created_at_and_dependency_versions():
    model, _ = get_fitted_model()
    metadata = SklearnSerializer().serialize(model)["metadata"]

    # Raises if not a valid ISO 8601 timestamp.
    datetime.fromisoformat(metadata["created_at"])

    assert metadata["dependency_versions"] == {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


def test_metadata_packages_single_package():
    model, _ = get_fitted_model()
    metadata = SklearnSerializer().serialize(model)["metadata"]

    assert metadata["packages"] == {"sklearn": sklearn.__version__}
    assert "producers" not in metadata


def test_metadata_packages_multiple_packages():
    class DummyThirdPartyTransformer(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            self.n_features_in_ = X.shape[1]
            return self

        def transform(self, X):
            return X

    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    y = np.array([0, 1, 1, 0])
    pipeline = Pipeline(
        [("dummy", DummyThirdPartyTransformer()), ("clf", LogisticRegression())]
    )
    pipeline.fit(X, y)

    serializer = SklearnSerializer(
        custom_estimators={"DummyThirdPartyTransformer": DummyThirdPartyTransformer}
    )
    packages = serializer.serialize(pipeline)["metadata"]["packages"]

    expected_third_party_name = DummyThirdPartyTransformer.__module__.split(".")[0]
    assert packages["sklearn"] == sklearn.__version__
    assert expected_third_party_name in packages


class DummyThirdPartyRegressor(BaseEstimator):
    def fit(self, X, y=None):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


def get_fitted_third_party_root():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    serializer = SklearnSerializer(
        custom_estimators={"DummyThirdPartyRegressor": DummyThirdPartyRegressor}
    )
    return DummyThirdPartyRegressor().fit(X, y), X, serializer


def test_metadata_producer_is_openmodels():
    model, _ = get_fitted_model()
    metadata = SklearnSerializer().serialize(model)["metadata"]

    # ONNX-style: the producer is the tool that wrote the file, not the model's package.
    assert metadata["producer_name"] == "openmodels"
    assert metadata["producer_version"] == _openmodels_version()
    # Superseded by producer_version in v3.
    assert "openmodels_version" not in metadata
    assert metadata["domain"] == "sklearn"
    assert metadata["domain_version"] == sklearn.__version__


def test_metadata_third_party_root_packages():
    model, _, serializer = get_fitted_third_party_root()
    metadata = serializer.serialize(model)["metadata"]

    package_name = DummyThirdPartyRegressor.__module__.split(".")[0]
    assert metadata["producer_name"] == "openmodels"
    assert package_name in metadata["packages"]
    # The domain package is always listed, even with no sklearn class in the tree.
    assert metadata["packages"]["sklearn"] == sklearn.__version__


def test_version_check_reads_domain_version():
    model, _ = get_fitted_model()
    serialized = SklearnSerializer().serialize(model)

    serialized["metadata"]["domain_version"] = "0.0.1"
    with pytest.warns(UserWarning, match="Version mismatch"):
        SklearnSerializer().deserialize(serialized)


def test_version_check_skips_unknown_domain_version():
    model, _ = get_fitted_model()
    serialized = SklearnSerializer().serialize(model)
    serialized["metadata"]["domain_version"] = "unknown"

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        SklearnSerializer().deserialize(serialized)


def test_v3_producer_version_never_compared_to_sklearn():
    # A v3 file from another writer (e.g. an R exporter) that doesn't record the scikit-learn
    # version: its producer_version is that tool's own version, not scikit-learn's.
    model, X, serializer = get_fitted_third_party_root()
    serialized = serializer.serialize(model)
    metadata = serialized["metadata"]
    metadata.pop("domain_version")
    metadata["producer_name"] = "my_r_exporter"
    metadata["producer_version"] = "1.2.0"

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        deserialized = serializer.deserialize(serialized)
    assert np.array_equal(model.predict(X), deserialized.predict(X))


def test_v2_metadata_falls_back_to_producer_version():
    model, _ = get_fitted_model()
    serialized = SklearnSerializer().serialize(model)
    metadata = serialized["metadata"]
    metadata.pop("domain_version")
    metadata["openmodels_format_version"] = 2
    metadata["producer_version"] = "0.0.1"

    with pytest.warns(UserWarning, match="scikit-learn 0.0.1"):
        SklearnSerializer().deserialize(serialized)


def test_package_version_mismatch_warns(monkeypatch):
    model, _, serializer = get_fitted_third_party_root()
    serialized = serializer.serialize(model)
    serialized["metadata"]["packages"]["somepkg"] = "1.0.0"
    monkeypatch.setattr(
        SklearnSerializer, "_resolve_package_version", staticmethod(lambda name: "2.0.0")
    )

    with pytest.warns(UserWarning, match="package 'somepkg'"):
        serializer.deserialize(serialized)


def test_v2_producers_map_still_checked(monkeypatch):
    model, _, serializer = get_fitted_third_party_root()
    serialized = serializer.serialize(model)
    metadata = serialized["metadata"]
    metadata["producers"] = {**metadata.pop("packages"), "somepkg": "1.0.0"}
    metadata["openmodels_format_version"] = 2
    monkeypatch.setattr(
        SklearnSerializer, "_resolve_package_version", staticmethod(lambda name: "2.0.0")
    )

    with pytest.warns(UserWarning, match="package 'somepkg'"):
        serializer.deserialize(serialized)
