import inspect
import warnings

import pytest
from sklearn.base import BaseEstimator

from openmodels.serializers.sklearn import _custom_estimator
from openmodels.serializers.sklearn.sklearn_serializer import SklearnSerializer


class DummyEstimator:
    pass


class NotEstimator:
    pass


# ==== is_valid_estimator ====

def test_is_valid_estimator_true():
    from sklearn.base import BaseEstimator
    class MyEstimator(BaseEstimator):
        pass
    assert _custom_estimator.is_valid_estimator("MyEstimator", MyEstimator)


def test_is_valid_estimator_false():
    assert not _custom_estimator.is_valid_estimator(123, DummyEstimator)
    assert not _custom_estimator.is_valid_estimator("NotEstimator", NotEstimator)


def test_is_valid_estimator_not_a_class():
    # name ok, but cls is a lambda or None
    assert not _custom_estimator.is_valid_estimator("func", lambda x: x)
    assert not _custom_estimator.is_valid_estimator("none", None)


def test_is_valid_estimator_typeerror():
    # cls is an instance, triggering TypeError inside issubclass
    class Fake:
        pass
    fake_instance = Fake()
    assert not _custom_estimator.is_valid_estimator("FakeInstance", fake_instance)


# ==== normalize_estimators ====

def test_normalize_estimators_list():
    ests = [DummyEstimator, DummyEstimator]
    result = _custom_estimator.normalize_estimators(ests)
    assert result == ests


def test_normalize_estimators_single():
    est = DummyEstimator
    result = _custom_estimator.normalize_estimators(est)
    assert result == [est]


def test_normalize_estimators_tuple_and_set():
    ests_tuple = (DummyEstimator, DummyEstimator)
    ests_set = {DummyEstimator}
    assert _custom_estimator.normalize_estimators(ests_tuple) == list(ests_tuple)
    assert set(_custom_estimator.normalize_estimators(ests_set)) == ests_set


# ==== load_custom_estimators ====

def test_load_custom_estimators_dict():
    from sklearn.base import BaseEstimator
    class MyEstimator(BaseEstimator):
        pass
    custom = {"MyEstimator": MyEstimator}
    all_estimators = {}
    result = _custom_estimator.load_custom_estimators(custom, all_estimators)
    assert "MyEstimator" in result
    assert result["MyEstimator"] is MyEstimator


def test_load_custom_estimators_callable():
    from sklearn.base import BaseEstimator
    class MyEstimator(BaseEstimator):
        pass
    def custom_callable():
        return [("MyEstimator", MyEstimator)]
    all_estimators = {}
    result = _custom_estimator.load_custom_estimators(custom_callable, all_estimators)
    assert "MyEstimator" in result
    assert result["MyEstimator"] is MyEstimator


def test_load_custom_estimators_invalid():
    def bad_callable():
        return [("Bad", NotEstimator)]
    all_estimators = {}
    result = _custom_estimator.load_custom_estimators(bad_callable, all_estimators)
    assert result == {}


def test_load_custom_estimators_callable_raises():
    # Covers line 36-38: callable raises exception
    def bad_callable():
        raise ValueError("fail")
    all_estimators = {}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = _custom_estimator.load_custom_estimators(bad_callable, all_estimators)
        assert result == {}
        assert any("Failed to call" in str(x.message) for x in w)


def test_load_custom_estimators_callable_returns_none():
    # Covers line 41: callable returns None
    def none_callable():
        return None
    all_estimators = {}
    result = _custom_estimator.load_custom_estimators(none_callable, all_estimators)
    assert result == {}


def test_load_custom_estimators_item_unpack_fail():
    # Covers lines 51-55: bad iterable unpacking
    def weird_callable():
        return [["badformat"]]
    all_estimators = {}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = _custom_estimator.load_custom_estimators(weird_callable, all_estimators)
        assert result == {}
        assert any("Unexpected" in str(x.message) for x in w)


def test_load_custom_estimators_conflict_warning():
    # Covers line 61: estimator name conflict
    from sklearn.base import BaseEstimator
    class MyEstimator(BaseEstimator):
        pass
    def custom_callable():
        return [("MyEstimator", MyEstimator)]
    all_estimators = {"MyEstimator": object}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = _custom_estimator.load_custom_estimators(custom_callable, all_estimators)
        assert "MyEstimator" in result
        assert any("conflicts" in str(x.message) for x in w)

def test_is_valid_estimator_typeerror_branch(monkeypatch):
    # Force inspect.isclass to return True so code reaches issubclass
    monkeypatch.setattr(inspect, "isclass", lambda x: True)
    # Passing an instance so issubclass(...) raises TypeError
    class Dummy: pass
    dummy_instance = Dummy()
    result = _custom_estimator.is_valid_estimator("BadEstimator", dummy_instance)
    assert result is False

# ==== (name, class) pairs ====


class PairEstimatorA(BaseEstimator):
    pass


class PairEstimatorB(BaseEstimator):
    pass


def _pair_sources():
    return [("PairEstimatorB", PairEstimatorB)]


def _registered(serializer):
    return {name: cls for name, cls in serializer._by_name.items() if cls in (PairEstimatorA, PairEstimatorB)}


def _no_warnings(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return fn()


@pytest.mark.parametrize(
    "custom",
    [
        [("PairEstimatorA", PairEstimatorA), ("PairEstimatorB", PairEstimatorB)],
        (("PairEstimatorA", PairEstimatorA), ("PairEstimatorB", PairEstimatorB)),
        [("PairEstimatorA", PairEstimatorA), _pair_sources],
        [["PairEstimatorA", PairEstimatorA], {"PairEstimatorB": PairEstimatorB}],
    ],
    ids=["list_of_pairs", "tuple_of_pairs", "pair_and_callable", "list_pair_and_dict"],
)
def test_serializer_registers_pairs(custom):
    serializer = _no_warnings(lambda: SklearnSerializer(custom_estimators=custom))
    assert _registered(serializer) == {"PairEstimatorA": PairEstimatorA, "PairEstimatorB": PairEstimatorB}


def test_serializer_registers_single_pair():
    serializer = _no_warnings(lambda: SklearnSerializer(custom_estimators=("PairEstimatorA", PairEstimatorA)))
    assert _registered(serializer) == {"PairEstimatorA": PairEstimatorA}


@pytest.mark.parametrize(
    "custom",
    [
        _pair_sources,
        {"PairEstimatorB": PairEstimatorB},
        [{"PairEstimatorA": PairEstimatorA}, {"PairEstimatorB": PairEstimatorB}],
        [lambda: [("PairEstimatorA", PairEstimatorA)], _pair_sources],
    ],
    ids=["callable", "dict", "list_of_dicts", "list_of_callables"],
)
def test_existing_source_forms_still_register(custom):
    serializer = _no_warnings(lambda: SklearnSerializer(custom_estimators=custom))
    expected = {"PairEstimatorB": PairEstimatorB}
    if not callable(custom) and not isinstance(custom, dict):
        expected["PairEstimatorA"] = PairEstimatorA
    assert _registered(serializer) == expected


def test_normalize_estimators_wraps_pairs():
    pair_a = ("PairEstimatorA", PairEstimatorA)
    assert _custom_estimator.normalize_estimators(pair_a) == [[pair_a]]
    assert _custom_estimator.normalize_estimators([pair_a, _pair_sources]) == [[pair_a], _pair_sources]


def test_pair_with_invalid_class_is_skipped_silently():
    result = _no_warnings(lambda: _custom_estimator.load_custom_estimators([("Bad", NotEstimator)], {}))
    assert result == {}


@pytest.mark.parametrize("element", [42, ["badformat"]], ids=["not_iterable", "not_a_pair"])
def test_malformed_element_warns(element):
    with pytest.warns(UserWarning, match="Unexpected custom_estimator format"):
        result = _custom_estimator.load_custom_estimators([("PairEstimatorA", PairEstimatorA), element], {})
    assert result == {"PairEstimatorA": PairEstimatorA}
