from openmodels.serializers.base import SerializerMixin


def test_str_keyed_dict_wire_shape_unchanged():
    mixin = SerializerMixin()
    value = {"a": 1, "b": 2}
    serialized = mixin.convert_to_serializable(value)
    assert serialized == value


def test_non_string_keyed_dict_round_trips():
    mixin = SerializerMixin()
    value = {1: "one", 2: "two", 3: "three"}
    serialized = mixin.convert_to_serializable(value)
    restored = mixin.convert_from_serializable(serialized, "dict")
    assert restored == value
    assert {type(k) for k in restored} == {int}


def test_mixed_key_type_dict_round_trips():
    mixin = SerializerMixin()
    value = {1: "a", "x": "b", 2.5: "c"}
    serialized = mixin.convert_to_serializable(value)
    restored = mixin.convert_from_serializable(serialized, "dict")
    assert restored == value
    assert {type(k) for k in restored} == {int, str, float}
