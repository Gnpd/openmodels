import sys

import pytest

from openmodels.converters.msgpack_converter import MsgpackConverter
from openmodels.converters.yaml_converter import YAMLConverter

SAMPLE = {
    "estimator_class": "LogisticRegression",
    "params": {"C": 1.0, "class_weight": None, "fit_intercept": True},
    "attributes": {"coef_": [[1.0, -2.0, 0.5]]},
}


def test_msgpack_round_trip():
    encoded = MsgpackConverter.serialize_to_format(SAMPLE)
    assert isinstance(encoded, bytes)
    assert MsgpackConverter.deserialize_from_format(encoded) == SAMPLE


def test_msgpack_rejects_non_dict():
    with pytest.raises(TypeError):
        MsgpackConverter.serialize_to_format("not a dict")


def test_msgpack_missing_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "msgpack", None)
    with pytest.raises(ImportError, match=r"pip install openmodels\[msgpack\]"):
        MsgpackConverter.serialize_to_format(SAMPLE)
    with pytest.raises(ImportError, match=r"pip install openmodels\[msgpack\]"):
        MsgpackConverter.deserialize_from_format(b"")


def test_yaml_round_trip():
    encoded = YAMLConverter.serialize_to_format(SAMPLE)
    assert isinstance(encoded, str)
    assert YAMLConverter.deserialize_from_format(encoded) == SAMPLE


def test_yaml_rejects_non_dict():
    with pytest.raises(TypeError):
        YAMLConverter.serialize_to_format("not a dict")


def test_yaml_preserves_field_order():
    encoded = YAMLConverter.serialize_to_format(SAMPLE)
    # sort_keys=False: "estimator_class" (first key) must appear before "attributes" (last key).
    assert encoded.index("estimator_class") < encoded.index("attributes")


def test_yaml_uses_safe_loader_only():
    # A !!python/object payload would construct an arbitrary object under yaml.load() with the
    # default Loader; safe_load must refuse it instead of executing it.
    malicious = "!!python/object/apply:os.system ['echo pwned']"
    with pytest.raises(yaml_constructor_error()):
        YAMLConverter.deserialize_from_format(malicious)


def yaml_constructor_error():
    import yaml

    return yaml.constructor.ConstructorError


def test_yaml_missing_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "yaml", None)
    with pytest.raises(ImportError, match=r"pip install openmodels\[yaml\]"):
        YAMLConverter.serialize_to_format(SAMPLE)
    with pytest.raises(ImportError, match=r"pip install openmodels\[yaml\]"):
        YAMLConverter.deserialize_from_format("")
