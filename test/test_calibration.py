import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from openmodels.core import SerializationManager
from openmodels.serializers.sklearn.sklearn_serializer import SklearnSerializer

X, y = make_classification(n_samples=60, n_features=4, random_state=0)


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
@pytest.mark.parametrize("format_name", ["json", "pickle"])
def test_calibrated_classifier_roundtrips(method, format_name):
    model = CalibratedClassifierCV(LogisticRegression(), cv=2, method=method).fit(X, y)
    manager = SerializationManager(SklearnSerializer())
    loaded = manager.deserialize(manager.serialize(model, format_name), format_name)
    assert len(loaded.calibrated_classifiers_) == 2
    np.testing.assert_array_equal(loaded.predict_proba(X), model.predict_proba(X))


def test_calibrators_load_through_deserialize_core(monkeypatch):
    """The public, root-only deserialize() (version checks, per-load state reset) runs once
    per load, not once more for every calibrator."""
    model = CalibratedClassifierCV(LogisticRegression(), cv=2).fit(X, y)
    serializer = SklearnSerializer()
    data = serializer.serialize(model)

    calls = []
    original = SklearnSerializer.deserialize

    def counting_deserialize(self, data):
        calls.append(data)
        return original(self, data)

    monkeypatch.setattr(SklearnSerializer, "deserialize", counting_deserialize)
    loaded = serializer.deserialize(data)
    assert len(calls) == 1
    np.testing.assert_array_equal(loaded.predict_proba(X), model.predict_proba(X))
