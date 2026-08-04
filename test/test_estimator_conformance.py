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
    ("Birch", "check_fit_score_takes_y"): (
        "openmodels serializes Birch's fitted state attribute-by-attribute, but root_/"
        "dummy_leaf_ are a custom linked tree of _CFNode/_CFSubcluster objects, not plain "
        "arrays; round-tripping loses the tree structure and predict() then fails with "
        "'need at least one array to concatenate'. Real gap - Birch isn't properly supported "
        "by the current sparse-tree-unaware serialization, tracked for a follow-up fix."
    ),
    ("SimpleImputer", "check_estimators_dtypes"): (
        "Real gap: SimpleImputer's private _fit_dtype attribute (a numpy dtype object, listed "
        "in ATTRIBUTE_EXCEPTIONS so openmodels does serialize it) round-trips through JSON as "
        "its string repr but is never reconstructed back into an np.dtype on deserialize, so "
        "transform() later fails with 'str' object has no attribute 'kind'. Passes at baseline "
        "with roundtrip_fit disabled - tracked for a follow-up fix to the dtype (de)serializer."
    ),
    ("OrdinalEncoder", "check_estimators_pickle"): (
        "Real gap: OrdinalEncoder's private _missing_indices attribute is a dict[int, int] "
        "(listed in ATTRIBUTE_EXCEPTIONS). JSON object keys must be strings, so after a "
        "round-trip the keys deserialize as str instead of int, and transform() later fails "
        "indexing X_int[:, cat_idx] with a string. Passes at baseline with roundtrip_fit "
        "disabled - tracked for a follow-up fix (int-keyed dicts need explicit key coercion "
        "on deserialize, the same class of issue as the tuple-vs-list JSON limitation)."
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
    ("BisectingKMeans", "check_estimators_dtypes"): (
        "Real gap: fitted cluster-center arrays kept as float32 get widened to float64 by the "
        "generic JSON round-trip (JSON has one numeric type), so predict()'s Cython inner loop "
        "later raises 'Buffer dtype mismatch, expected const float but got double'. Passes at "
        "baseline with roundtrip_fit disabled - tracked for a follow-up fix (dtype needs to be "
        "preserved/restored explicitly for estimators that fit in float32)."
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
            "KernelRidge": ["check_sample_weight_equivalence_on_sparse_data"],
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

# Real, scoped openmodels gap surfaced by this suite: the sparse (de)serializer in
# openmodels/serializers/base.py only registers a handler for scipy.sparse.csr_matrix
# (ScipySerializerMixin._get_serializer_handlers), so csc_matrix and the newer scipy/sklearn
# array-API sparse containers (csr_array, csc_array, ...) raise "not JSON serializable" instead
# of round-tripping. sklearn 1.9's check_estimator machinery increasingly exercises estimators
# with csr_array. Only bites estimators that actually declare + exercise sparse support in these
# specific checks - most estimators correctly reject/ignore sparse input before ever reaching
# serialization, so this is listed per-estimator (confirmed via the strict xfail mechanism
# below) rather than as a blanket check-level skip. Worth a real fix (broaden the handler to
# scipy.sparse.issparse(), not just csr_matrix) as a follow-up - out of scope for this change.
_SPARSE_GAP_REASON = (
    "openmodels's sparse (de)serializer only handles scipy.sparse.csr_matrix; "
    "csc_matrix/csr_array/csc_array are not yet supported."
)
KNOWN_ROUNDTRIP_XFAILS.update(
    _entries(
        _SPARSE_GAP_REASON,
        {
            "AffinityPropagation": [
                "check_estimator_sparse_tag",
                "check_estimator_sparse_array",
            ],
            "DBSCAN": ["check_estimator_sparse_tag", "check_estimator_sparse_array"],
            "Isomap": ["check_estimator_sparse_tag", "check_estimator_sparse_array"],
            "KNeighborsClassifier": [
                "check_estimator_sparse_tag",
                "check_estimator_sparse_array",
            ],
            "KNeighborsRegressor": [
                "check_estimator_sparse_tag",
                "check_estimator_sparse_array",
            ],
            "KNeighborsTransformer": [
                "check_estimator_sparse_tag",
                "check_estimator_sparse_array",
            ],
            "KernelPCA": ["check_estimator_sparse_tag", "check_estimator_sparse_array"],
            "KernelRidge": [
                "check_estimator_sparse_tag",
                "check_estimator_sparse_array",
                "check_estimator_sparse_matrix",
            ],
            "LabelPropagation": [
                "check_estimator_sparse_tag",
                "check_estimator_sparse_array",
                "check_estimator_sparse_matrix",
            ],
            "NearestNeighbors": [
                "check_estimator_sparse_tag",
                "check_estimator_sparse_array",
            ],
            "Nystroem": ["check_estimator_sparse_tag", "check_estimator_sparse_array"],
            "RadiusNeighborsClassifier": [
                "check_estimator_sparse_tag",
                "check_estimator_sparse_array",
            ],
            "RadiusNeighborsRegressor": [
                "check_estimator_sparse_tag",
                "check_estimator_sparse_array",
            ],
        },
    )
)


def _check_name(check) -> str:
    return getattr(check, "func", check).__name__


@parametrize_with_checks(ESTIMATORS)
def test_estimator_conformance_roundtrip(estimator, check, request):
    name = _check_name(check)
    reason = KNOWN_ROUNDTRIP_XFAILS.get((type(estimator).__name__, name))

    # Applied dynamically (rather than pytest.xfail(), which would abort here without ever
    # running the check) so that strict=True can catch the check actually starting to pass -
    # e.g. once one of the real gaps documented above gets fixed. Without strict mode, a fixed
    # bug would keep reporting XFAIL forever and nobody would be prompted to clean up the entry.
    if reason is not None:
        request.applymarker(pytest.mark.xfail(reason=reason, strict=True))

    with roundtrip_fit(type(estimator)):
        check(estimator)
