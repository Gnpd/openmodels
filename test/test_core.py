import json

import numpy as np
import pytest
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer
from openmodels.core import SerializationManager
from openmodels.serializers.sklearn.sklearn_serializer import SklearnSerializer
from openmodels.exceptions import (
    SerializationError,
    DeserializationError,
    UnsupportedEstimatorError,
    UnsupportedFormatError,
)


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


def test_save_and_load_logistic_regression_msgpack(tmp_path):
    model, X = get_fitted_model()
    manager = SerializationManager(SklearnSerializer())
    file_path = tmp_path / "model.msgpack"
    manager.save(model, file_path, format_name="msgpack")
    assert file_path.exists()
    loaded_model = manager.load(file_path, format_name="msgpack")
    assert hasattr(loaded_model, "predict")
    assert np.array_equal(model.predict(X), loaded_model.predict(X))


def test_save_and_load_logistic_regression_yaml(tmp_path):
    model, X = get_fitted_model()
    manager = SerializationManager(SklearnSerializer())
    file_path = tmp_path / "model.yaml"
    manager.save(model, file_path, format_name="yaml")
    assert file_path.exists()
    loaded_model = manager.load(file_path, format_name="yaml")
    assert hasattr(loaded_model, "predict")
    assert np.array_equal(model.predict(X), loaded_model.predict(X))


def test_save_requires_file_path():
    model, _ = get_fitted_model()
    manager = SerializationManager(SklearnSerializer())
    with pytest.raises(TypeError):
        manager.save(model, format_name="json")


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
        def serialize(self, model):
            return "not a dict"

        def deserialize(self, data):
            return "not a model"

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

    def bad_open(*args, **kwargs):
        raise IOError("fail")

    monkeypatch.setattr("builtins.open", bad_open)
    with pytest.raises(SerializationError):
        manager.save(model, "model.json", format_name="json")


def test_load_file_io_error(monkeypatch, tmp_path):
    manager = SerializationManager(SklearnSerializer())
    file_path = tmp_path / "model.json"
    file_path.write_text("{}")

    def bad_open(*args, **kwargs):
        raise IOError("fail")

    monkeypatch.setattr("builtins.open", bad_open)
    with pytest.raises(DeserializationError):
        manager.load(file_path, format_name="json")


# ==== every manager failure is an OpenModels error ====


class _Unencodable:
    pass


def test_serialize_non_estimator_raises_serialization_error():
    manager = SerializationManager(SklearnSerializer())
    with pytest.raises(SerializationError) as exc_info:
        manager.serialize(object())
    assert isinstance(exc_info.value.__cause__, AttributeError)


def test_serialize_unencodable_param_raises_serialization_error():
    manager = SerializationManager(SklearnSerializer())
    model = FunctionTransformer(kw_args={"w": _Unencodable()})
    with pytest.raises(SerializationError, match="not JSON serializable") as exc_info:
        manager.serialize(model, format_name="json")
    assert isinstance(exc_info.value.__cause__, TypeError)


def test_deserialize_malformed_model_raises_deserialization_error():
    manager = SerializationManager(SklearnSerializer())
    with pytest.raises(DeserializationError, match="estimator_class") as exc_info:
        manager.deserialize('{"params": {}}')
    assert isinstance(exc_info.value.__cause__, KeyError)


def test_deserialize_invalid_json_keeps_cause():
    manager = SerializationManager(SklearnSerializer())
    with pytest.raises(DeserializationError) as exc_info:
        manager.deserialize("not a valid json")
    assert isinstance(exc_info.value.__cause__, json.JSONDecodeError)


def test_unknown_estimator_class_passes_through():
    model, _ = get_fitted_model()
    manager = SerializationManager(SklearnSerializer())
    data = json.loads(manager.serialize(model))
    data["estimator_class"] = "Nope"
    with pytest.raises(UnsupportedEstimatorError, match="Nope"):
        manager.deserialize(json.dumps(data))


def test_crafted_function_reference_is_not_double_wrapped():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 1])
    manager = SerializationManager(SklearnSerializer())
    data = json.loads(manager.serialize(SelectKBest(chi2, k=1).fit(X, y)))
    data["params"]["score_func"] = {"module": "os", "name": "getcwd"}
    with pytest.raises(DeserializationError) as exc_info:
        manager.deserialize(json.dumps(data))
    assert str(exc_info.value).startswith("function 'os.getcwd' is not allowed")
    assert exc_info.value.__cause__ is None


def test_save_unserializable_model_raises_serialization_error(tmp_path):
    manager = SerializationManager(SklearnSerializer())
    file_path = tmp_path / "model.json"
    with pytest.raises(SerializationError):
        manager.save(object(), file_path, format_name="json")
    assert not file_path.exists()


def test_load_malformed_file_raises_deserialization_error(tmp_path):
    manager = SerializationManager(SklearnSerializer())
    file_path = tmp_path / "model.json"
    file_path.write_text('{"params": {}}')
    with pytest.raises(DeserializationError) as exc_info:
        manager.load(file_path, format_name="json")
    assert isinstance(exc_info.value.__cause__, KeyError)
