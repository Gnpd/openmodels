"""
Tuple-valued params: every format stores them as lists, and scikit-learn's param validation
rejects a list for params declared tuple-only, so a loaded model couldn't be fitted again
(clone() in cross-validation, retraining a loaded pipeline, ...). They're turned back into
tuples on load when the estimator's _parameter_constraints require one.
"""

import json

import numpy as np
import pytest
from sklearn.base import BaseEstimator, clone
from sklearn.cluster import SpectralBiclustering
from sklearn.feature_extraction.image import PatchExtractor
from sklearn.feature_extraction.text import (
    CountVectorizer,
    HashingVectorizer,
    TfidfVectorizer,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from openmodels.core import SerializationManager
from openmodels.serializers.sklearn.sklearn_serializer import SklearnSerializer

rng = np.random.RandomState(0)
X = rng.rand(30, 4)
DOCS = [f"doc {i} word{i % 5} token{i % 3}" for i in range(30)]
IMAGES = rng.rand(3, 10, 10)

# (estimator, tuple-only param, data to fit on)
CASES = {
    "MinMaxScaler": (MinMaxScaler(feature_range=(-1, 1)), "feature_range", X),
    "RobustScaler": (RobustScaler(), "quantile_range", X),
    "CountVectorizer": (CountVectorizer(ngram_range=(1, 2)), "ngram_range", DOCS),
    "TfidfVectorizer": (TfidfVectorizer(), "ngram_range", DOCS),
    "HashingVectorizer": (HashingVectorizer(), "ngram_range", DOCS),
    "PatchExtractor": (PatchExtractor(patch_size=(3, 3)), "patch_size", IMAGES),
    "SpectralBiclustering": (
        SpectralBiclustering(n_clusters=(2, 2), random_state=0),
        "n_clusters",
        X,
    ),
}


def _roundtrip(model, format_name="json", serializer=None):
    manager = SerializationManager(serializer or SklearnSerializer())
    return manager.deserialize(manager.serialize(model, format_name), format_name)


@pytest.mark.parametrize("format_name", ["json", "pickle"])
@pytest.mark.parametrize("name", CASES)
def test_tuple_param_restored_and_refittable(name, format_name):
    model, param, data = CASES[name]
    model = clone(model).fit(data)
    loaded = _roundtrip(model, format_name)
    restored = getattr(loaded, param)
    assert isinstance(restored, tuple) and restored == getattr(model, param)
    loaded.fit(data)  # param validation used to reject the list
    clone(loaded).fit(data)


def test_array_like_params_still_validate():
    """Params that also accept lists/array-likes are left as they come back."""
    loaded = _roundtrip(MLPClassifier(hidden_layer_sizes=(3,)))
    loaded._validate_params()


class _TupleParamEstimator(BaseEstimator):
    """A third-party-style estimator: a tuple param and no _parameter_constraints."""

    def __init__(self, sizes=(1, 2)):
        self.sizes = sizes

    def fit(self, X=None, y=None):
        self.fitted_ = True
        return self


class _NoValidationEstimator(BaseEstimator):
    _parameter_constraints = {"sizes": "no_validation"}

    def __init__(self, sizes=(1, 2)):
        self.sizes = sizes

    def fit(self, X=None, y=None):
        self.fitted_ = True
        return self


@pytest.mark.parametrize("cls", [_TupleParamEstimator, _NoValidationEstimator])
def test_params_without_tuple_constraints_are_unchanged(cls):
    serializer = SklearnSerializer(custom_estimators={cls.__name__: cls})
    data = serializer.serialize(cls().fit())
    loaded = SerializationManager(serializer).deserialize(json.dumps(data))
    assert loaded.sizes == [1, 2]
