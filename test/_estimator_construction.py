"""
Shared source of truth for "what does it take to construct this estimator at all" - mostly
meta-estimators that wrap another estimator or composite transformers that require
sub-components, which raise TypeError if built with bare defaults.

This is used two ways:
- test_estimator_conformance.py imports CONSTRUCTOR_ARGS/construct() wholesale: it only needs
  *unfitted* instances (sklearn's own check_estimator machinery fits them with its own
  synthetic data), so the minimal entries here are sufficient on their own.
- test_classification.py/test_regression.py/test_transformation.py import individual
  CONSTRUCTOR_ARGS[name] entries (via BASE_CLASSIFIER/BASE_REGRESSOR or the dict directly) for
  the subset of estimators where their own special-casing needs nothing more than "a valid
  estimator to wrap" - avoiding maintaining the same "this class needs an estimator= kwarg"
  fact independently in multiple files (see the sklearn-1.9-compat branch's audit notes: this
  already caused HalvingGridSearchCV support to be added in only one of two files that needed
  it). Where a file's fitting data legitimately calls for a richer choice than the minimal
  default here (e.g. StackingRegressor/VotingRegressor mixing in a RandomForestRegressor for
  more realistic coverage, or ColumnTransformer/FeatureUnion needing column selectors matched
  to real data), that file keeps its own local, richer construction instead of using this
  registry - this file only holds the bare minimum needed to construct, not every reasonable
  choice.

This is deliberately separate from the fit-data special-casing in test_regression.py,
test_classification.py, etc. - those shape input data for this repo's own smoke tests, which
is a different concern from what's required to construct the estimator at all.
"""

import inspect
from typing import Any, Dict

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.multioutput import ClassifierChain, RegressorChain
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_CLASSIFIER = LogisticRegression(solver="lbfgs")
BASE_REGRESSOR = LinearRegression()


def _chain_estimator_kwarg(cls: type) -> str:
    """ClassifierChain/RegressorChain renamed their wrapped-estimator constructor
    parameter from base_estimator to estimator in scikit-learn 1.7 (deprecating the old
    name, and it's gone entirely by 1.9) - detect which one this installed scikit-learn
    actually accepts instead of hardcoding a version cutoff, so this keeps working however
    the deprecation timeline shifts across the versions this repo tests against."""
    params = inspect.signature(cls.__init__).parameters
    return "estimator" if "estimator" in params else "base_estimator"


CONSTRUCTOR_ARGS: Dict[str, Dict[str, Any]] = {
    "ClassifierChain": {_chain_estimator_kwarg(ClassifierChain): BASE_CLASSIFIER},
    "FixedThresholdClassifier": {"estimator": BASE_CLASSIFIER},
    "TunedThresholdClassifierCV": {"estimator": BASE_CLASSIFIER},
    "OneVsOneClassifier": {"estimator": BASE_CLASSIFIER},
    "OneVsRestClassifier": {"estimator": BASE_CLASSIFIER},
    "OutputCodeClassifier": {"estimator": BASE_CLASSIFIER},
    "SelfTrainingClassifier": {"estimator": BASE_CLASSIFIER},
    "MultiOutputClassifier": {"estimator": BASE_CLASSIFIER},
    "StackingClassifier": {
        "estimators": [
            ("lr1", BASE_CLASSIFIER),
            ("lr2", LogisticRegression(solver="lbfgs", random_state=1)),
        ]
    },
    "VotingClassifier": {
        "estimators": [
            ("lr1", BASE_CLASSIFIER),
            ("lr2", LogisticRegression(solver="lbfgs", random_state=1)),
        ]
    },
    "MultiOutputRegressor": {"estimator": BASE_REGRESSOR},
    "RegressorChain": {_chain_estimator_kwarg(RegressorChain): BASE_REGRESSOR},
    "StackingRegressor": {
        "estimators": [("lr1", BASE_REGRESSOR), ("lr2", LinearRegression())]
    },
    "VotingRegressor": {
        "estimators": [("lr1", BASE_REGRESSOR), ("lr2", LinearRegression())]
    },
    "SelectFromModel": {"estimator": BASE_CLASSIFIER},
    "SequentialFeatureSelector": {"estimator": BASE_CLASSIFIER},
    "RFE": {"estimator": BASE_CLASSIFIER},
    "RFECV": {"estimator": BASE_CLASSIFIER},
    "ColumnTransformer": {
        "transformers": [
            ("num", StandardScaler(), [0, 1]),
            ("cat", OneHotEncoder(), [2]),
        ]
    },
    "FeatureUnion": {"transformer_list": [("scaler", StandardScaler())]},
    "GridSearchCV": {
        "estimator": LogisticRegression(solver="lbfgs"),
        "param_grid": {"C": [1.0]},
    },
    "RandomizedSearchCV": {
        "estimator": LogisticRegression(solver="lbfgs"),
        "param_distributions": {"C": [1.0]},
        "n_iter": 1,
    },
    "Pipeline": {"steps": [("clf", LogisticRegression(solver="lbfgs"))]},
    # Default hyperparameters that are incompatible with the tiny synthetic datasets
    # sklearn's own check_estimator machinery generates internally (unrelated to openmodels
    # round-trip fidelity - these would fail identically with no serialization involved).
    "SparseRandomProjection": {"n_components": 2},
    "GaussianRandomProjection": {"n_components": 2},
    "TSNE": {"perplexity": 5},
    "CCA": {"n_components": 1},
    "PLSCanonical": {"n_components": 1},
    "PLSSVD": {"n_components": 1},
}

# Composite/meta transformers whose valid construction and/or expected input shape is too
# specific to the estimator (column selectors, dictionaries sized to a particular n_features,
# text-only input, y-only fit/transform) to be exercised meaningfully by sklearn's generic,
# single-2D-X check battery. They're still covered by this repo's own smoke tests
# (test_transformation.py), which know how to build fitting data for them.
NOT_CHECKED: Dict[str, str] = {
    "ColumnTransformer": "column selectors are tied to a specific input shape/dtype",
    "FeatureUnion": "sub-transformers are tied to a specific input shape",
    "SparseCoder": "dictionary must be sized to match n_features",
    "DictVectorizer": "expects list-of-dicts input, not a 2D array",
    "HashingVectorizer": "expects text input, not a 2D array",
    "FeatureHasher": "expects dicts/strings input, not a 2D array",
    "LabelBinarizer": "fits/transforms on y only, not X",
    "LabelEncoder": "fits/transforms on y only, not X",
    "MultiLabelBinarizer": "fits/transforms on label collections, not a 2D array",
    "FrozenEstimator": "wraps an already-fitted estimator; fit() is a no-op by design",
    "SpectralBiclustering": "n_best/n_components/n_clusters interact and need per-check tuning to fit tiny synthetic data - not an openmodels round-trip issue",
    "SpectralCoclustering": "n_best/n_components/n_clusters interact and need per-check tuning to fit tiny synthetic data - not an openmodels round-trip issue",
    "HalvingGridSearchCV": "successive-halving resource scheduling needs more samples than check_estimator's tiny synthetic datasets provide - not an openmodels round-trip issue",
    "HalvingRandomSearchCV": "successive-halving resource scheduling needs more samples than check_estimator's tiny synthetic datasets provide - not an openmodels round-trip issue",
}


def construct(cls: type) -> Any:
    """Instantiate `cls` with its default constructor, or the required kwargs above."""
    return cls(**CONSTRUCTOR_ARGS.get(cls.__name__, {}))
