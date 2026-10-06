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


def normalize_estimators(
    estimators: Union[Callable[..., Any], List[Any], Tuple[Any, ...], Dict[str, Any]],
) -> List[Any]:
    """Normalize input into a flat list of estimators or (name, class) items."""
    if not isinstance(estimators, (list, tuple, set)):
        return [estimators]
    return list(estimators)


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

        if isinstance(items, dict):
            iterator = items.items()
        else:
            iterator = items

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
