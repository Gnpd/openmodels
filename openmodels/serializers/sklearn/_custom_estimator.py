import inspect
import warnings
from typing import Any, Callable, Dict, Iterator, List, Tuple, Type, Union


def is_valid_estimator(name: str, cls: Any) -> bool:
    """Check whether (name, cls) represents a valid sklearn estimator."""

    if not isinstance(name, str):
        return False
    if not inspect.isclass(cls):
        return False
    try:
        from sklearn.base import BaseEstimator

        return issubclass(cls, BaseEstimator)
    except TypeError:
        return False


def _is_pair(item: Any) -> bool:
    """Whether `item` has the shape of one (name, class) pair rather than a source of pairs.
    No source (a callable, a dict or an iterable of pairs) starts with a string, so a
    2-item tuple/list starting with one is a pair; its class is validated later."""
    return (
        isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[0], str)
    )


def normalize_estimators(
    estimators: Union[Callable[..., Any], List[Any], Tuple[Any, ...], Dict[str, Any]],
) -> List[Any]:
    """Normalize input into a flat list of sources, each a callable, a dict or an iterable of
    (name, class) pairs. A single pair, or a pair given as an element of a list of sources, is
    wrapped into a one-pair source of its own."""
    if _is_pair(estimators):
        return [[estimators]]
    if not isinstance(estimators, (list, tuple, set)):
        return [estimators]
    return [[est] if _is_pair(est) else est for est in estimators]


def iter_custom_estimators(
    custom_estimators: Union[
        Callable[..., Any], List[Any], Tuple[Any, ...], Dict[str, Any]
    ],
) -> Iterator[Tuple[str, Type]]:
    """Yield every valid (name, class) pair from user-provided estimators, in order and
    without merging pairs that share a name."""
    for est in normalize_estimators(custom_estimators):
        try:
            items = est() if callable(est) else est
        except Exception:
            warnings.warn("Failed to call custom_estimator(); skipping.", UserWarning)
            continue

        if items is None:
            continue

        try:
            iterator = iter(items.items() if isinstance(items, dict) else items)
        except TypeError:
            warnings.warn("Unexpected custom_estimator format; skipping.", UserWarning)
            continue

        for item in iterator:
            try:
                name, cls = item
            except Exception:
                warnings.warn(
                    "Unexpected custom_estimator format; skipping.", UserWarning
                )
                continue

            if not is_valid_estimator(name, cls):
                continue

            yield name, cls


def load_custom_estimators(
    custom_estimators: Union[
        Callable[..., Any], List[Any], Tuple[Any, ...], Dict[str, Any]
    ],
    all_estimators: Dict[str, Type],
) -> Dict[str, Type]:
    """Convert user-provided estimators into a dictionary of valid ones."""
    extra = {}
    for name, cls in iter_custom_estimators(custom_estimators):
        if name in all_estimators and all_estimators[name] is not cls:
            warnings.warn(
                f"Estimator '{name}' conflicts with built-in one; the custom version wins "
                f"only for files without estimator_package (format v1-v3). Files written by "
                f"openmodels >= 0.2.3 record each class's package and keep both.",
                UserWarning,
            )

        extra[name] = cls

    return extra
