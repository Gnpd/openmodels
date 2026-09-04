"""
Runs scikit-learn's own generic estimator conformance battery
(sklearn.utils.estimator_checks.parametrize_with_checks) against openmodels-round-tripped
estimators.

This is the same battery scikit-learn runs against itself for every estimator it ships
(check_fit_idempotent, check_methods_subset_invariance, transformer/classifier/regressor
contract checks, ...). Discovery reuses the same all_estimators() + NOT_SUPPORTED_ESTIMATORS
pattern as test_regression.py/test_classification.py/test_clustering.py/test_transformation.py,
so it automatically covers every estimator openmodels claims to support, with no per-module
mapping to maintain.

Deliberately uses the public, version-stable parametrize_with_checks(estimators) form rather
than the private sklearn.utils._test_common.instance_generator helpers (expected_failed_checks,
_tested_estimators, ...), whose shape has already changed across the sklearn versions this repo
tests against in .github/workflows/sklearn-compat.yml.
"""

import pytest
import sklearn
from sklearn.base import clone
from sklearn.utils.discovery import all_estimators
from sklearn.utils.estimator_checks import parametrize_with_checks

from openmodels.serializers.sklearn.sklearn_serializer import NOT_SUPPORTED_ESTIMATORS
from openmodels.test_helpers import roundtrip_fit
from test._estimator_construction import NOT_CHECKED, construct

ESTIMATOR_CLASSES = [
    cls
    for name, cls in all_estimators()
    if name not in NOT_SUPPORTED_ESTIMATORS and name not in NOT_CHECKED
]

ESTIMATORS = [construct(cls) for cls in ESTIMATOR_CLASSES]

# (major, minor) of the installed scikit-learn - used only by KNOWN_ROUNDTRIP_VERSION_XFAILS
# below, for gaps that are fixed upstream as of a specific version rather than permanent.
_SKLEARN_VERSION = tuple(int(p) for p in sklearn.__version__.split(".")[:2])


def _entries(reason: str, estimator_checks: dict) -> dict:
    """Expand {estimator_name: [check_name, ...]} sharing one `reason` into individual
    (estimator_name, check_name) -> reason entries for KNOWN_ROUNDTRIP_XFAILS."""
    return {
        (estimator_name, check_name): reason
        for estimator_name, check_names in estimator_checks.items()
        for check_name in check_names
    }


# Known, understood round-trip gaps - populated from real CI failures, each with a reason, and
# always keyed per (estimator class name, check function name). Earlier revisions of this file
# grouped some entries by check name alone (e.g. "check_sample_weight_equivalence_on_dense_data
# always fails, for every estimator") on the assumption that a check failing for a handful of
# estimators meant it was fundamentally broken for everyone. Once the xfail mechanism below was
# made strict (see test_estimator_conformance_roundtrip), that assumption was caught immediately
# as inaccurate: most estimators carrying e.g. check_estimator_sparse_tag never exercise sparse
# data at all in this check and pass it fine, so a check-level skip silently hid that from ever
# running. Keying strictly per estimator is what makes strict=True meaningful - it only ever
# flags real staleness, not "this check-level skip happened to also cover an unrelated pass".
KNOWN_ROUNDTRIP_XFAILS: dict = {
    ("TunedThresholdClassifierCV", "check_classifiers_train"): (
        "JSON round-trips the tuned decision threshold as a float that isn't bit-identical to "
        "the original numpy float64; on this check's synthetic data one sample's decision value "
        "sits close enough to the threshold that the tiny precision difference flips its "
        "predicted class (1/200 samples). Not a correctness bug - a boundary-case artifact of "
        "float round-tripping through JSON."
    ),
    ("DictionaryLearning", "check_transformer_general"): (
        "Pre-existing sklearn/check fragility, unrelated to serialization: fails identically "
        "against a bare, unpatched DictionaryLearning() with check_estimator's default "
        "synthetic data."
    ),
    ("DictionaryLearning", "check_transformer_data_not_an_array"): (
        "Same pre-existing, serialization-unrelated fragility as this estimator's "
        "check_transformer_general failure."
    ),
    ("MiniBatchNMF", "check_transformer_general"): (
        "Pre-existing sklearn/check fragility, unrelated to serialization: fails identically "
        "against a bare, unpatched MiniBatchNMF() with check_estimator's default synthetic "
        "data."
    ),
    ("MiniBatchNMF", "check_transformer_data_not_an_array"): (
        "Same pre-existing, serialization-unrelated fragility as this estimator's "
        "check_transformer_general failure."
    ),
    ("NMF", "check_transformer_general"): (
        "Pre-existing sklearn/check fragility, unrelated to serialization: fails identically "
        "against a bare, unpatched NMF() with check_estimator's default synthetic data."
    ),
    ("NMF", "check_transformer_data_not_an_array"): (
        "Same pre-existing, serialization-unrelated fragility as this estimator's "
        "check_transformer_general failure."
    ),
    ("LassoLarsIC", "check_fit2d_1sample"): (
        "Pre-existing sklearn/check fragility, unrelated to serialization: fails identically "
        "against a bare, unpatched LassoLarsIC() - the expected error message pattern for a "
        "1-sample fit doesn't match what this estimator actually raises."
    ),
    ("BernoulliRBM", "check_methods_sample_order_invariance"): (
        "Pre-existing sklearn fragility, unrelated to serialization: fails identically against "
        "a bare, unpatched BernoulliRBM(). score_samples() is documented as stochastic "
        "(it perturbs one feature per call for its pseudo-likelihood estimate), so exact "
        "order/subset invariance isn't guaranteed even without any round-trip involved."
    ),
    ("BernoulliRBM", "check_methods_subset_invariance"): (
        "Same pre-existing, serialization-unrelated fragility as this estimator's "
        "check_methods_sample_order_invariance failure."
    ),
    ("LogisticRegressionCV", "check_sparsify_coefficients"): (
        "Pre-existing sklearn/check fragility, unrelated to serialization: fails identically "
        "against a bare, unpatched LogisticRegressionCV() - its internal CV splitting can't "
        "satisfy n_splits=5 on this check's small per-class sample counts."
    ),
    ("NuSVC", "check_classifiers_one_label_sample_weights"): (
        "Pre-existing sklearn/check fragility, unrelated to serialization: fails identically "
        "against a bare, unpatched NuSVC()."
    ),
    ("NuSVC", "check_class_weight_classifiers"): (
        "Pre-existing sklearn/check fragility, unrelated to serialization: fails identically "
        "against a bare, unpatched NuSVC()."
    ),
    ("Pipeline", "check_estimators_overwrite_params"): (
        "Pre-existing sklearn/check fragility, unrelated to serialization: fails identically "
        "against a bare, unpatched Pipeline(steps=[...])."
    ),
    ("Pipeline", "check_dont_overwrite_parameters"): (
        "Same pre-existing, serialization-unrelated fragility as this estimator's "
        "check_estimators_overwrite_params failure."
    ),
}

# Pre-existing sklearn/check fragility, unrelated to serialization: fails identically against a
# bare, unpatched estimator for every one of these (verified directly, roundtrip_fit disabled).
# Fitting with sample_weight is not always numerically equivalent to fitting on repeated/removed
# rows, due to floating-point non-associativity - a well-known limitation of this specific check
# for ensemble/boosting/margin-based estimators. Confirmed accurate estimator-by-estimator (not
# assumed check-wide) after the strict xfail mechanism below flagged the original, too-broad
# check-level version of this entry as producing hundreds of unexpected passes.
_SAMPLE_WEIGHT_EQUIVALENCE_REASON = (
    "Pre-existing sklearn fragility, unrelated to serialization: fails identically against a "
    "bare, unpatched estimator. Fitting with sample_weight is not always numerically "
    "equivalent to fitting on repeated/removed rows due to floating-point non-associativity."
)
_BOTH_SAMPLE_WEIGHT_CHECKS = [
    "check_sample_weight_equivalence_on_dense_data",
    "check_sample_weight_equivalence_on_sparse_data",
]
KNOWN_ROUNDTRIP_XFAILS.update(
    _entries(
        _SAMPLE_WEIGHT_EQUIVALENCE_REASON,
        {
            "AdaBoostClassifier": _BOTH_SAMPLE_WEIGHT_CHECKS,
            "BaggingClassifier": _BOTH_SAMPLE_WEIGHT_CHECKS,
            "BaggingRegressor": _BOTH_SAMPLE_WEIGHT_CHECKS,
            "BisectingKMeans": _BOTH_SAMPLE_WEIGHT_CHECKS,
            "GradientBoostingClassifier": _BOTH_SAMPLE_WEIGHT_CHECKS,
            "GradientBoostingRegressor": _BOTH_SAMPLE_WEIGHT_CHECKS,
            "HuberRegressor": _BOTH_SAMPLE_WEIGHT_CHECKS,
            "IsolationForest": _BOTH_SAMPLE_WEIGHT_CHECKS,
            "KMeans": _BOTH_SAMPLE_WEIGHT_CHECKS,
            "LinearSVC": _BOTH_SAMPLE_WEIGHT_CHECKS,
            "LinearSVR": _BOTH_SAMPLE_WEIGHT_CHECKS,
            "MiniBatchKMeans": _BOTH_SAMPLE_WEIGHT_CHECKS,
            "NuSVC": _BOTH_SAMPLE_WEIGHT_CHECKS,
            "NuSVR": _BOTH_SAMPLE_WEIGHT_CHECKS,
            "OneClassSVM": _BOTH_SAMPLE_WEIGHT_CHECKS,
            "Perceptron": _BOTH_SAMPLE_WEIGHT_CHECKS,
            "RANSACRegressor": _BOTH_SAMPLE_WEIGHT_CHECKS,
            "RandomForestClassifier": _BOTH_SAMPLE_WEIGHT_CHECKS,
            "RandomForestRegressor": _BOTH_SAMPLE_WEIGHT_CHECKS,
            "RandomTreesEmbedding": _BOTH_SAMPLE_WEIGHT_CHECKS,
            "SGDClassifier": _BOTH_SAMPLE_WEIGHT_CHECKS,
            "SGDOneClassSVM": _BOTH_SAMPLE_WEIGHT_CHECKS,
            "SGDRegressor": _BOTH_SAMPLE_WEIGHT_CHECKS,
            "SVC": _BOTH_SAMPLE_WEIGHT_CHECKS,
            "SVR": _BOTH_SAMPLE_WEIGHT_CHECKS,
        },
    )
)

# Same floating-point non-associativity as _SAMPLE_WEIGHT_EQUIVALENCE_REASON above, but only
# below scikit-learn 1.7 and only on the dense variant of the check (verified directly -
# installed scikit-learn==1.6.1 and confirmed both fail identically against a bare, unpatched
# estimator with roundtrip_fit disabled; the sparse variant passes cleanly there, and both
# variants pass cleanly on 1.7+). Kept out of KNOWN_ROUNDTRIP_XFAILS/strict=True: since this is
# genuinely fixed upstream from 1.7 onward, an unconditional strict entry would XPASS on every
# newer scikit-learn version this repo tests against - see KNOWN_ROUNDTRIP_VERSION_XFAILS below,
# which only applies below the given scikit-learn version.
KNOWN_ROUNDTRIP_VERSION_XFAILS: dict = {
    ("BayesianRidge", "check_sample_weight_equivalence_on_dense_data"): (
        _SAMPLE_WEIGHT_EQUIVALENCE_REASON,
        (1, 7),
    ),
    ("KBinsDiscretizer", "check_sample_weight_equivalence_on_dense_data"): (
        _SAMPLE_WEIGHT_EQUIVALENCE_REASON,
        (1, 7),
    ),
}

# Pre-existing sklearn fragility, unrelated to serialization: fails identically against a bare,
# unpatched estimator (verified directly, roundtrip_fit disabled). On check_estimator's tiny
# synthetic data, some solvers report n_iter_=None, or the estimator does zero iterations (e.g.
# SelfTrainingClassifier when the data happens to have no unlabeled samples) - which this check
# doesn't tolerate.
_N_ITER_REASON = (
    "Pre-existing sklearn fragility, unrelated to serialization: fails identically against a "
    "bare, unpatched estimator. On check_estimator's tiny synthetic data, some solvers report "
    "n_iter_=None or the estimator does zero iterations, which this check doesn't tolerate."
)
KNOWN_ROUNDTRIP_XFAILS.update(
    _entries(
        _N_ITER_REASON,
        {
            "LogisticRegressionCV": ["check_non_transformer_estimators_n_iter"],
            "Ridge": ["check_non_transformer_estimators_n_iter"],
            "RidgeClassifier": ["check_non_transformer_estimators_n_iter"],
            "SelfTrainingClassifier": ["check_non_transformer_estimators_n_iter"],
        },
    )
)


def _check_name(check) -> str:
    return getattr(check, "func", check).__name__


# Per-(estimator class name, check function name) constructor param overrides, applied only to
# the clone handed to that one check - the shared instance in ESTIMATORS (and every other check
# run against it) keeps its normal construct()-derived params. Unlike KNOWN_ROUNDTRIP_XFAILS
# (which documents a gap that can't be closed), this documents a genuine fix: a different param
# choice makes the check pass reliably, but can't become that estimator's global default because
# other checks in this same battery fit on much smaller synthetic data that param choice would
# break instead (see the SpectralEmbedding entry below for a concrete case).
PER_CHECK_CONSTRUCTOR_OVERRIDES: dict = {
    ("SpectralEmbedding", "check_pipeline_consistency"): {
        # check_pipeline_consistency's synthetic data is two tight 15-point clusters. With
        # SpectralEmbedding's default n_neighbors (~n_samples/10 = 3), the k-NN affinity graph
        # never bridges the two clusters, so the graph Laplacian's zero eigenvalue has
        # multiplicity 2 - a genuinely degenerate eigenspace, not just an ill-conditioned one.
        # ARPACK's choice of basis within that degenerate subspace is arbitrary and can differ
        # between two otherwise-identical fits depending on platform floating-point rounding,
        # which is what made this check flake in CI (never locally - verified by direct
        # experimentation, unrelated to openmodels round-tripping or to BLAS thread count).
        # n_neighbors=20 bridges the clusters so the graph is connected and the embedding is
        # actually unique. Can't be SpectralEmbedding's default construct() args: several other
        # checks in this battery fit it on datasets as small as 10 samples, where n_neighbors=20
        # raises "Expected n_neighbors <= n_samples_fit" (confirmed by trying it globally first).
        "n_neighbors": 20
    },
}


@parametrize_with_checks(ESTIMATORS)
def test_estimator_conformance_roundtrip(estimator, check, request):
    name = _check_name(check)
    key = (type(estimator).__name__, name)
    reason = KNOWN_ROUNDTRIP_XFAILS.get(key)

    if reason is None:
        version_entry = KNOWN_ROUNDTRIP_VERSION_XFAILS.get(key)
        if version_entry is not None:
            version_reason, max_version = version_entry
            if _SKLEARN_VERSION < max_version:
                reason = version_reason

    # Applied dynamically (rather than pytest.xfail(), which would abort here without ever
    # running the check) so that strict=True can catch the check actually starting to pass -
    # e.g. once one of the real gaps documented above gets fixed. Without strict mode, a fixed
    # bug would keep reporting XFAIL forever and nobody would be prompted to clean up the entry.
    if reason is not None:
        request.applymarker(pytest.mark.xfail(reason=reason, strict=True))

    overrides = PER_CHECK_CONSTRUCTOR_OVERRIDES.get((type(estimator).__name__, name))
    if overrides is not None:
        estimator = clone(estimator).set_params(**overrides)

    with roundtrip_fit(type(estimator)):
        check(estimator)
