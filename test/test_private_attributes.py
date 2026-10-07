"""
Fitted state, mostly private attributes, that estimators need after loading for methods other
than predict/transform. Each used to be missing, so the method raised on the loaded copy while
predict kept working, which is all the smoke tests compare.
"""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.utils.discovery import all_estimators

from openmodels.core import SerializationManager
from openmodels.exceptions import DeserializationError, SerializationError
from openmodels.serializers.sklearn.sklearn_serializer import SklearnSerializer
from test._estimator_construction import construct

FORMATS = ["json", "pickle", "msgpack", "yaml"]

rng = np.random.RandomState(0)
X = rng.rand(80, 4) + 0.1  # positive: some transformers below need non-negative input
y = (X[:, 0] + X[:, 1] > 1.1).astype(int)
y3 = (X[:, 0] * 3).astype(int) % 3
yr = X @ np.array([1.0, -2.0, 0.5, 3.0])


def _roundtrip(model, format_name="json"):
    manager = SerializationManager(SklearnSerializer())
    return manager.deserialize(manager.serialize(model, format_name), format_name)


# ==== HistGradientBoostingClassifier._loss, HistGradientBoosting*._n_features ====


@pytest.mark.parametrize("format_name", FORMATS)
@pytest.mark.parametrize("target", [y, y3], ids=["binary", "multiclass"])
def test_hgb_classifier_predict_proba(target, format_name):
    model = HistGradientBoostingClassifier(max_iter=10).fit(X, target)
    loaded = _roundtrip(model, format_name)
    np.testing.assert_array_equal(loaded.predict_proba(X), model.predict_proba(X))
    for got, expected in zip(
        loaded.staged_predict_proba(X), model.staged_predict_proba(X)
    ):
        np.testing.assert_array_equal(got, expected)


def test_hgb_regressor_staged_predict():
    model = HistGradientBoostingRegressor(max_iter=10).fit(X, yr)
    loaded = _roundtrip(model)
    staged = list(loaded.staged_predict(X))
    assert len(staged) == 10
    np.testing.assert_array_equal(staged[-1], model.predict(X))


def test_soft_voting_with_hgb_classifier_predicts():
    model = VotingClassifier(
        [
            ("h", HistGradientBoostingClassifier(max_iter=10)),
            ("l", LogisticRegression()),
        ],
        voting="soft",
    ).fit(X, y)
    np.testing.assert_array_equal(_roundtrip(model).predict(X), model.predict(X))


# ==== LinearDiscriminantAnalysis._max_components ====


@pytest.mark.parametrize("n_components", [None, 1])
def test_lda_transform(n_components):
    model = LinearDiscriminantAnalysis(n_components=n_components).fit(X, y3)
    loaded = _roundtrip(model)
    assert loaded._max_components == model._max_components
    np.testing.assert_allclose(loaded.transform(X), model.transform(X))
    assert list(loaded.get_feature_names_out()) == list(model.get_feature_names_out())


# ==== _n_features_out (generic) and Stacking's _n_feature_outs ====

N_FEATURES_OUT_ESTIMATORS = [
    "BernoulliRBM",
    "Birch",
    "BisectingKMeans",
    "CCA",
    "FeatureAgglomeration",
    "GaussianRandomProjection",
    "Isomap",
    "KMeans",
    "KNeighborsTransformer",
    "LinearDiscriminantAnalysis",
    "LocallyLinearEmbedding",
    "MiniBatchKMeans",
    "Nystroem",
    "PLSCanonical",
    "PLSRegression",
    "PLSSVD",
    "PolynomialCountSketch",
    "RBFSampler",
    "RadiusNeighborsTransformer",
    "RandomTreesEmbedding",
    "SkewedChi2Sampler",
    "SparseRandomProjection",
    "StackingClassifier",
    "StackingRegressor",
]
_CLASSES = dict(all_estimators())


@pytest.mark.parametrize("name", N_FEATURES_OUT_ESTIMATORS)
def test_get_feature_names_out(name):
    model = construct(_CLASSES[name])
    target = (
        yr
        if name
        in ("CCA", "PLSCanonical", "PLSRegression", "PLSSVD", "StackingRegressor")
        else y
    )
    model.fit(X, target)
    loaded = _roundtrip(model)
    assert list(loaded.get_feature_names_out()) == list(model.get_feature_names_out())


def test_pandas_output_pipeline_predicts():
    model = make_pipeline(
        KMeans(n_clusters=3, n_init=1, random_state=0), LogisticRegression()
    ).fit(X, y)
    loaded = _roundtrip(model).set_output(transform="pandas")
    np.testing.assert_array_equal(loaded.predict(pd.DataFrame(X)), model.predict(X))


def test_n_features_out_saved_only_as_instance_attribute():
    """PCA computes _n_features_out in a property: saving it would make loading try to set a
    property. KMeans stores it on the instance, so it's saved."""
    serializer = SklearnSerializer()
    assert (
        "_n_features_out" in serializer.serialize(KMeans(n_init=1).fit(X))["attributes"]
    )
    pca = PCA(2).fit(X)
    assert "_n_features_out" not in serializer.serialize(pca)["attributes"]
    assert _roundtrip(pca)._n_features_out == 2


# ==== HDBSCAN._single_linkage_tree_ (a structured array) ====


@pytest.mark.parametrize("format_name", FORMATS)
def test_hdbscan_dbscan_clustering(format_name):
    model = HDBSCAN().fit(X)
    loaded = _roundtrip(model, format_name)
    assert loaded._single_linkage_tree_.dtype == model._single_linkage_tree_.dtype
    np.testing.assert_array_equal(
        loaded._single_linkage_tree_, model._single_linkage_tree_
    )
    np.testing.assert_array_equal(
        loaded.dbscan_clustering(0.3), model.dbscan_clustering(0.3)
    )


@pytest.mark.parametrize(
    "bad_dtype", ["[__import__('os').getcwd()]", "[('a', 'not-a-type')]", "[unclosed"]
)
def test_invalid_structured_dtype_is_refused(bad_dtype):
    data = SklearnSerializer().serialize(HDBSCAN().fit(X))
    data["attribute_dtypes"]["_single_linkage_tree_"] = bad_dtype
    with pytest.raises(DeserializationError, match="Invalid structured dtype"):
        SerializationManager(SklearnSerializer()).deserialize(json.dumps(data))


# ==== search estimators' scorer_, rebuilt from `scoring` ====

SCORINGS = {
    "default": None,
    "string": "neg_mean_absolute_error",
    "list": ["r2", "neg_mean_absolute_error"],
    "dict": {"fit": "r2", "error": "neg_max_error"},
}


@pytest.mark.parametrize("format_name", ["json", "pickle"])
@pytest.mark.parametrize("scoring", SCORINGS.values(), ids=SCORINGS.keys())
@pytest.mark.parametrize("search_cls", [GridSearchCV, RandomizedSearchCV])
def test_search_score(search_cls, scoring, format_name):
    refit = (
        scoring[0]
        if isinstance(scoring, list)
        else "error" if isinstance(scoring, dict) else True
    )
    model = search_cls(
        Ridge(), {"alpha": [0.1, 1.0]}, cv=2, scoring=scoring, refit=refit
    )
    if search_cls is RandomizedSearchCV:
        model.set_params(n_iter=2, random_state=0)
    model.fit(X, yr)
    loaded = _roundtrip(model, format_name)
    assert loaded.score(X, yr) == model.score(X, yr)
    assert type(loaded.scorer_) is type(model.scorer_)
    if isinstance(model.scorer_, dict):
        assert loaded.scorer_.keys() == model.scorer_.keys()


def test_search_scorer_is_never_saved():
    model = GridSearchCV(
        Ridge(),
        {"alpha": [0.1, 1.0]},
        cv=2,
        scoring=["r2", "neg_mean_absolute_error"],
        refit="r2",
    ).fit(X, yr)
    assert "scorer_" not in SklearnSerializer().serialize(model)["attributes"]


def test_search_with_scorer_object_as_scoring_still_refused():
    """The `scoring` param itself can't hold a scorer object (make_scorer(...)): saving fails
    cleanly, before scorer_ is involved."""
    model = GridSearchCV(
        Ridge(), {"alpha": [0.1, 1.0]}, cv=2, scoring=make_scorer(mean_absolute_error)
    ).fit(X, yr)
    with pytest.raises(SerializationError, match="Can't serialize callable"):
        SerializationManager(SklearnSerializer()).serialize(model)
