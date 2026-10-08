"""
Resuming training on a loaded model: for every supported estimator that can continue training
(`partial_fit`, or `warm_start=True` and fit again), continuing on the loaded copy must give the
same model as continuing on the original. Compared with every inference method, as in
test_inference_parity.py.

Loaded models are meant for inference; resuming works for most estimators because their
training state is part of their fitted state. KNOWN_FAILURES lists the ones whose running
training state (optimizer momentum, running sums, random generator position, ...) isn't saved.
They're strict xfails: once one is supported, the test asks for its entry to be removed.

The two models are fitted separately with the same seed, never copied: copy.deepcopy doesn't
keep NumPy views, which some estimators update through during training (Birch's CF-tree nodes),
so a deep copy of a fitted model can continue differently from the model itself.
"""

from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
from sklearn.base import is_classifier, is_regressor
from sklearn.linear_model import SGDClassifier, SGDRegressor

from openmodels.core import SerializationManager
from openmodels.serializers.sklearn.sklearn_serializer import (
    ALL_ESTIMATORS,
    SklearnSerializer,
)
from test import test_inference_parity as parity

pytestmark = pytest.mark.filterwarnings("ignore")

# Params grown between the two fits of a warm start, so the second fit does more work.
WARM_START_GROWN = ("max_iter", "n_estimators")

_NOT_SAVED = "running training state isn't saved: "
KNOWN_FAILURES: Dict[Tuple[str, str], str] = {
    ("MiniBatchDictionaryLearning", "partial_fit"): _NOT_SAVED
    + "_random_state, _A, _B, ...",
    ("MiniBatchKMeans", "partial_fit"): _NOT_SAVED
    + "_counts, _n_since_last_reassign, ...",
    ("MiniBatchNMF", "partial_fit"): _NOT_SAVED + "_components_numerator, ...",
    ("MLPClassifier", "partial_fit"): _NOT_SAVED
    + "_optimizer, _no_improvement_count, ...",
    ("MLPRegressor", "partial_fit"): _NOT_SAVED
    + "_optimizer, _no_improvement_count, ...",
    ("GradientBoostingClassifier", "warm_start"): _NOT_SAVED + "_rng",
    ("GradientBoostingRegressor", "warm_start"): _NOT_SAVED + "_rng",
    ("HistGradientBoostingClassifier", "warm_start"): _NOT_SAVED + "_random_seed, ...",
    ("HistGradientBoostingRegressor", "warm_start"): _NOT_SAVED + "_random_seed, ...",
}


def _construct(name: str) -> Any:
    """parity._construct, except that a meta-estimator whose partial_fit is only available
    when its wrapped estimator has one (OneVsRestClassifier, MultiOutputRegressor,
    SelectFromModel, ...) wraps an SGD model, so its partial_fit is tested too."""
    estimator = parity._construct(name)
    if hasattr(ALL_ESTIMATORS[name], "partial_fit") and not hasattr(
        estimator, "partial_fit"
    ):
        inner = SGDRegressor if is_regressor(estimator) else SGDClassifier
        estimator.set_params(estimator=inner(random_state=0))
    return estimator


def _resumable() -> List[Tuple[str, str]]:
    cases = []
    for name in parity.ESTIMATOR_NAMES:
        estimator = _construct(name)
        if hasattr(estimator, "partial_fit"):
            cases.append((name, "partial_fit"))
        if "warm_start" in estimator.get_params(deep=False):
            cases.append((name, "warm_start"))
    return cases


RESUMABLE = _resumable()


def _fit(name: str, kind: str) -> Tuple[Any, SimpleNamespace]:
    estimator = _construct(name)
    if kind == "warm_start":
        estimator.set_params(warm_start=True)
    data = parity._data_for(name, estimator)
    try:
        estimator.fit(*data.fit_args)
    except TypeError:  # fit(X) only
        estimator.fit(data.fit_args[0])
    return estimator, data


def _continue(estimator: Any, kind: str, data: SimpleNamespace) -> None:
    if kind == "partial_fit":
        kwargs = {}
        if is_classifier(estimator):
            y = np.asarray(data.y)
            kwargs["classes"] = (
                [np.unique(col) for col in y.T] if y.ndim == 2 else np.unique(y)
            )
        try:
            estimator.partial_fit(*data.fit_args, **kwargs)
        except TypeError:  # partial_fit(X) only
            estimator.partial_fit(data.fit_args[0])
        return
    params = estimator.get_params(deep=False)
    estimator.set_params(
        **{
            key: params[key] * 2
            for key in WARM_START_GROWN
            if isinstance(params.get(key), int) and not isinstance(params[key], bool)
        }
    )
    try:
        estimator.fit(*data.fit_args)
    except TypeError:
        estimator.fit(data.fit_args[0])


@pytest.mark.parametrize("format_name", parity.FORMATS)
@pytest.mark.parametrize(
    "name, kind", RESUMABLE, ids=[f"{name}-{kind}" for name, kind in RESUMABLE]
)
def test_resume_training(name, kind, format_name, request):
    reason = KNOWN_FAILURES.get((name, kind))
    if reason is not None:
        request.applymarker(pytest.mark.xfail(reason=reason, strict=True))

    original, data = _fit(name, kind)
    to_save, _ = _fit(name, kind)
    manager = SerializationManager(SklearnSerializer())
    loaded = manager.deserialize(manager.serialize(to_save, format_name), format_name)

    _continue(original, kind, data)
    _continue(loaded, kind, data)

    failures, compared = [], 0
    for method in parity.INFERENCE_CALLS:
        try:
            compared += parity._check_method(method, original, loaded, data)
        except Exception as e:
            failures.append(f"{method}: {type(e).__name__}: {str(e)[:300]}")
    assert (
        not failures
    ), f"{name} [{kind}, {format_name}] differs after resuming:\n  " + "\n  ".join(
        failures
    )
    assert compared, f"{name}: no inference method returned a result to compare"


def test_known_failures_are_real_cases():
    for case in KNOWN_FAILURES:
        assert case in RESUMABLE, case
