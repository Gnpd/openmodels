import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from openmodels.core import SerializationManager
from openmodels.serializers.sklearn.sklearn_serializer import SklearnSerializer
from openmodels.exceptions import SerializationError, DeserializationError, UnsupportedFormatError

def get_fitted_model():
    X = np.array([[0, 0], [1, 1], [2, 2]])
    y = np.array([0, 1, 1])
    model = LogisticRegression()
    model.fit(X, y)
    return model, X

def test_save_and_load_logistic_regression_json(tmp_path):
    model, X = get_fitted_model()
    manager = SerializationManager(SklearnSerializer())
    file_path = tmp_path / "model.json"
    manager.save(model, file_path, format_name="json")
    assert file_path.exists()
    loaded_model = manager.load(file_path, format_name="json")
    assert hasattr(loaded_model, "predict")
    assert np.array_equal(model.predict(X), loaded_model.predict(X))

def test_save_and_load_logistic_regression_pickle(tmp_path):
    model, X = get_fitted_model()
    manager = SerializationManager(SklearnSerializer())
    file_path = tmp_path / "model.pkl"
    manager.save(model, file_path, format_name="pickle")
    assert file_path.exists()
    loaded_model = manager.load(file_path, format_name="pickle")
    assert hasattr(loaded_model, "predict")
    assert np.array_equal(model.predict(X), loaded_model.predict(X))

def test_save_with_unsupported_format(tmp_path):
    model, _ = get_fitted_model()
    manager = SerializationManager(SklearnSerializer())
    file_path = tmp_path / "model.unsupported"
    with pytest.raises(UnsupportedFormatError):
        manager.save(model, file_path, format_name="unsupported")

def test_load_with_unsupported_format(tmp_path):
    manager = SerializationManager(SklearnSerializer())
    file_path = tmp_path / "model.unsupported"
    file_path.write_text("dummy")
    with pytest.raises(UnsupportedFormatError):
        manager.load(file_path, format_name="unsupported")

def test_save_with_bad_serializer(tmp_path):
    class BadSerializer:
        def serialize(self, model): return "not a dict"
        def deserialize(self, data): return "not a model"
    manager = SerializationManager(BadSerializer())
    file_path = tmp_path / "model.json"
    with pytest.raises(SerializationError):
        manager.save({}, file_path, format_name="json")

def test_load_with_bad_data(tmp_path):
    manager = SerializationManager(SklearnSerializer())
    file_path = tmp_path / "model.json"
    file_path.write_text("not a valid json")
    with pytest.raises(DeserializationError):
        manager.load(file_path, format_name="json")

def test_save_file_io_error(monkeypatch):
    model, _ = get_fitted_model()
    manager = SerializationManager(SklearnSerializer())
    def bad_open(*args, **kwargs): raise IOError("fail")
    monkeypatch.setattr("builtins.open", bad_open)
    with pytest.raises(SerializationError):
        manager.save(model, "model.json", format_name="json")

def test_load_file_io_error(monkeypatch, tmp_path):
    manager = SerializationManager(SklearnSerializer())
    file_path = tmp_path / "model.json"
    file_path.write_text("{}")
    def bad_open(*args, **kwargs): raise IOError("fail")
    monkeypatch.setattr("builtins.open", bad_open)
    with pytest.raises(DeserializationError):
        manager.load(file_path, format_name="json")