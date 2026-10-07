from openmodels.serializers.base import SerializerMixin


def test_str_keyed_dict_wire_shape_unchanged():
    mixin = SerializerMixin()
    value = {"a": 1, "b": 2}
    serialized = mixin.convert_to_serializable(value)
    assert serialized == value


def test_non_string_keyed_dict_round_trips():
    """Non-string keys are written as text; their types come from the dict's type entry."""
    mixin = SerializerMixin()
    value = {1: "one", 2: "two", 3: "three"}
    serialized = mixin.convert_to_serializable(value)
    assert serialized == {"1": "one", "2": "two", "3": "three"}
    value_type = {
        "dict": {"1": "str", "2": "str", "3": "str"},
        "key_types": {"1": "int", "2": "int", "3": "int"},
    }
    restored = mixin.convert_from_serializable(serialized, value_type)
    assert restored == value
    assert {type(k) for k in restored} == {int}


def test_mixed_key_type_dict_round_trips():
    mixin = SerializerMixin()
    value = {1: "a", "x": "b", 2.5: "c"}
    serialized = mixin.convert_to_serializable(value)
    value_type = {
        "dict": {"1": "str", "x": "str", "2.5": "str"},
        "key_types": {"1": "int", "2.5": "float"},
    }
    restored = mixin.convert_from_serializable(serialized, value_type)
    assert restored == value
    assert {type(k) for k in restored} == {int, str, float}


def test_envelope_dict_from_older_files_round_trips():
    """Files written before format v4 saved non-string-keyed dicts as an envelope."""
    mixin = SerializerMixin()
    envelope = {
        "__openmodels_dict__": True,
        "keys": [1, "x", 2.5],
        "key_types": ["int", "str", "float"],
        "values": ["a", "b", "c"],
    }
    restored = mixin.convert_from_serializable(envelope, "dict")
    assert restored == {1: "a", "x": "b", 2.5: "c"}
    assert {type(k) for k in restored} == {int, str, float}
