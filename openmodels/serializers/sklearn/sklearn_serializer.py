"""
Scikit-learn model serializer for the OpenModels library.

This module provides a serializer for scikit-learn models, allowing them to be
converted to and from dictionary representations.
"""

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Set,
    Tuple,
    Type,
    Optional,
    Union,
)
from importlib.metadata import version as _package_version, PackageNotFoundError
from datetime import datetime, timezone
import platform
import sys
import numpy as np
import scipy  # type: ignore
import inspect
from scipy.sparse import issparse  # type: ignore

from ._custom_estimator import iter_custom_estimators, load_custom_estimators

import sklearn
from sklearn.calibration import _CalibratedClassifier, _SigmoidCalibration
from sklearn.compose import make_column_selector
from sklearn.cluster._birch import _CFNode, _CFSubcluster
from sklearn.cluster._bisect_k_means import _BisectingTree
from sklearn.ensemble._hist_gradient_boosting.predictor import TreePredictor
from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
from sklearn.gaussian_process import kernels as _gp_kernels
from sklearn.gaussian_process.kernels import Kernel
from sklearn.gaussian_process._gpc import _BinaryGaussianProcessClassifierLaplace
from sklearn._loss.loss import (
    AbsoluteError,
    HalfBinomialLoss,
    HalfGammaLoss,
    HalfMultinomialLoss,
    HalfPoissonLoss,
    HalfSquaredError,
    HalfTweedieLoss,
    HalfTweedieLossIdentity,
    PinballLoss,
    BaseLoss,
)
from sklearn.metrics._scorer import _CurveScorer, _MultimetricScorer
from sklearn.metrics import get_scorer_names, get_scorer
from sklearn.multiclass import _ConstantPredictor
from sklearn.tree._tree import Tree
from sklearn.base import BaseEstimator, check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.utils import Bunch
from sklearn.utils.discovery import all_estimators
from sklearn.neighbors import BallTree, KDTree, KernelDensity
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

# Private, but its tree-building code (NeighborsBase._fit) is identical across TESTED_VERSIONS;
# see _rebuild_neighbors_tree.
from sklearn.neighbors._base import NeighborsBase

# Private, but `_get_scorers` (how fit builds scorer_ from `scoring`) is identical across
# TESTED_VERSIONS; see _rebuild_derived_attributes.
from sklearn.model_selection._search import BaseSearchCV

from openmodels.exceptions import DeserializationError, UnsupportedEstimatorError
from openmodels.protocols import ModelSerializer
from openmodels.serializers.base import (
    NumpySerializerMixin,
    ScipySerializerMixin,
    _key_to_text,
)
import warnings

ConverterFunc = Callable[[Any], Any]

# Fitted attributes a neighbors estimator's search tree is rebuilt from on load.
_NEIGHBORS_TREE_INPUTS = (
    "_fit_X",
    "_fit_method",
    "effective_metric_",
    "effective_metric_params_",
)

LOSS_CLASS_REGISTRY = {
    "AbsoluteError": AbsoluteError,
    "HalfBinomialLoss": HalfBinomialLoss,
    "HalfGammaLoss": HalfGammaLoss,
    "HalfMultinomialLoss": HalfMultinomialLoss,
    "HalfPoissonLoss": HalfPoissonLoss,
    "HalfSquaredError": HalfSquaredError,
    "HalfTweedieLoss": HalfTweedieLoss,
    "HalfTweedieLossIdentity": HalfTweedieLossIdentity,
    "PinballLoss": PinballLoss,
}

ALL_ESTIMATORS = {
    name: cls for name, cls in all_estimators() if issubclass(cls, BaseEstimator)
}
# add extra private estimators to ALL_ESTIMATORS
ALL_ESTIMATORS["_BinMapper"] = _BinMapper
ALL_ESTIMATORS["_SigmoidCalibration"] = _SigmoidCalibration
ALL_ESTIMATORS["_BinaryGaussianProcessClassifierLaplace"] = (
    _BinaryGaussianProcessClassifierLaplace
)
ALL_ESTIMATORS["_ConstantPredictor"] = _ConstantPredictor

TESTED_VERSIONS = ["1.6.1", "1.7.2", "1.8.0", "1.9.1"]

# Version of openmodels's own wire format (the shape of the serialized dict), independent of
# scikit-learn's version (domain_version) and of the writing tool's release version
# (producer_version, informational only). Bump this only when the structure or the meaning of
# its fields changes.
# v2: producer_version/producer_name/domain/openmodels_format_version/openmodels_version moved
# from flat top-level keys into a single nested "metadata" dict, present once at the true root
# (not duplicated on every nested/composite sub-estimator as in v1).
# v3: producer_name/producer_version follow ONNX and name the tool that wrote the file
# ("openmodels", or e.g. an R exporter), not the outermost model's package + scikit-learn's
# version; scikit-learn's version moved to the new domain_version field, and "producers" was
# renamed "packages" (always including the domain package). Readers fall back to
# producer_version as the scikit-learn version, and to "producers", for v1/v2 files only.
# openmodels_version was dropped: producer_version holds the same value.
# v4: every estimator node (root and nested) records "estimator_package" (the class's top-level
# package) next to "estimator_class", and classes are resolved by (package, class name), so
# same-named classes from different packages coexist. Files without it (v1-v3) resolve by bare
# name, custom estimators winning. Also in v4: a dict whose values need restoring (arrays,
# tuples, estimators, ...) is typed per key, as {"dict": {key: type}} ({"Bunch": ...} for a
# scikit-learn Bunch), with a matching {key: dtype} dtypes entry; a tuple inside it is typed
# {"tuple": [types]}, and a masked array (cv_results_'s param_* columns) "MaskedArray". Non-string
# keys (e.g. class_weight={0: 1.0}) are written as text, with their types in a "key_types"
# entry next to "dict"; v1-v3 files wrote such dicts as an `__openmodels_dict__` envelope,
# which is still read. Dicts of plain JSON values with string keys keep the "dict" tag. v3
# readers still load v4 files but don't know these types: values load as their saved JSON,
# and non-string keys as strings.
OPENMODELS_FORMAT_VERSION = 4

# Type tags whose saved values every format gives back unchanged: a dict whose values all have
# one of these (or are lists of them) needs no per-key types and keeps the "dict" tag.
_PLAIN_TYPES = frozenset({"str", "int", "float", "bool", "NoneType"})


def _is_plain_type(tag: Any) -> bool:
    if isinstance(tag, str):
        return tag in _PLAIN_TYPES
    if isinstance(tag, list):
        return all(_is_plain_type(t) for t in tag)
    return False


def _package_of(cls: type) -> str:
    """Top-level package of a class (e.g. "sklearn", "chemotools", "__main__"). The full
    module path isn't used: scikit-learn's real modules are private and renamed between
    releases."""
    return cls.__module__.split(".")[0]


def _openmodels_version() -> str:
    """Installed openmodels release version, read from package metadata so pyproject.toml
    stays the single source of truth. Falls back to "unknown" for editable/from-source
    installs without proper metadata."""
    try:
        return _package_version("openmodels")
    except PackageNotFoundError:
        return "unknown"


NOT_SUPPORTED_ESTIMATORS: list[str] = [
    # Regressors: all regressors work!! Hurray!
    # Classifiers: all classifiers work!! Hurray!
    # Clusters: all clusters work!! Hurray!
    # Transformers: all transformers work!! Hurray!
    # Others: all others work!! Hurray!
]


# Dictionary of attribute exceptions
ATTRIBUTE_EXCEPTIONS: Dict[str, List] = {
    # Regressors:
    "PLSRegression": ["_x_mean", "_x_std", "_y_mean", "_y_std", "_predict_1d"],
    "SVR": [
        "_sparse",
        "_n_support",
        "_dual_coef_",
        "_intercept_",
        "_probA",
        "_probB",
        "_gamma",
        "_effective_probability",
    ],
    "KNeighborsRegressor": ["_fit_method", "_fit_X", "_y"],
    "NuSVR": [
        "_sparse",
        "_gamma",
        "_n_support",
        "_probA",
        "_probB",
        "_dual_coef_",
        "_intercept_",
        "_effective_probability",
    ],
    "TweedieRegressor": ["_base_loss"],
    "GaussianProcessRegressor": ["kernel_", "_y_train_std", "_y_train_mean"],
    "GradientBoostingRegressor": ["_loss"],
    "HistGradientBoostingRegressor": [
        "_loss",
        "_preprocessor",
        "_baseline_prediction",
        "_predictors",
        "_bin_mapper",
    ],
    "RadiusNeighborsRegressor": ["_fit_method", "_fit_X", "_y"],
    "CCA": ["_x_mean", "_x_std", "_y_mean", "_y_std", "_predict_1d"],
    "GammaRegressor": ["_base_loss"],
    "PoissonRegressor": ["_base_loss"],
    "PLSCanonical": ["_x_mean", "_x_std", "_y_mean", "_y_std", "_predict_1d"],
    "IsotonicRegression": ["f_"],
    "TransformedTargetRegressor": ["_training_dim"],
    "StackingRegressor": ["_n_feature_outs"],
    # Clusters:
    "BisectingKMeans": ["_bisecting_tree", "_n_threads", "_X_mean"],
    "Birch": ["_subcluster_norms"],
    "HDBSCAN": ["_single_linkage_tree_"],
    "KMeans": ["_n_threads"],
    "MiniBatchKMeans": ["_n_threads"],
    # Classifiers:
    "_BinaryGaussianProcessClassifierLaplace": ["kernel_"],
    "DummyClassifier": ["_strategy"],
    "HistGradientBoostingClassifier": [
        "_preprocessor",
        "_baseline_prediction",
        "_predictors",
        "_bin_mapper",
    ],
    "GradientBoostingClassifier": ["_loss"],
    "MLPClassifier": ["_label_binarizer"],
    "NuSVC": [
        "_sparse",
        "_n_support",
        "_probA",
        "_probB",
        "_gamma",
        "_dual_coef_",
        "_intercept_",
        "_effective_probability",
    ],
    "KNeighborsClassifier": ["_fit_method", "_fit_X", "_y", "_tree"],
    "RadiusNeighborsClassifier": ["_fit_method", "_fit_X", "_y", "_tree"],
    "RidgeClassifier": ["_label_binarizer"],
    "RidgeClassifierCV": ["_label_binarizer"],
    "StackingClassifier": ["_label_encoder", "_n_feature_outs"],
    "SVC": [
        "_sparse",
        "_n_support",
        "_dual_coef_",
        "_intercept_",
        "_probA",
        "_probB",
        "_gamma",
        "_effective_probability",
    ],
    "TunedThresholdClassifierCV": ["_curve_scorer"],
    # Transformers:
    "ColumnTransformer": ["_columns", "_remainder", "_transformer_to_input_indices"],
    "OneHotEncoder": [
        "_infrequent_enabled",
        "_drop_idx_after_grouping",
        "_n_features_outs",
    ],
    "OrdinalEncoder": ["_missing_indices", "_infrequent_enabled"],
    "KBinsDiscretizer": ["_encoder"],
    "KernelPCA": ["_centerer"],
    "KNNImputer": ["_fit_X", "_mask_fit_X", "_valid_mask"],
    "KNeighborsTransformer": ["_fit_method", "_tree", "_fit_X"],
    "PowerTransformer": ["_scaler"],
    "RadiusNeighborsTransformer": ["_fit_method", "_tree", "_fit_X"],
    "SimpleImputer": ["_fit_dtype", "_fill_dtype"],
    "MiniBatchNMF": ["_n_components", "_transform_max_iter", "_beta_loss", "_gamma"],
    "NMF": ["_n_components", "_beta_loss"],
    "MissingIndicator": ["_n_features", "_precomputed"],
    "MultiLabelBinarizer": ["_cached_dict"],
    "PolynomialFeatures": ["_max_degree", "_n_out_full", "_min_degree"],
    "PLSSVD": ["_x_mean", "_x_std", "_y_mean", "_y_std"],
    "TargetEncoder": ["_infrequent_enabled"],
    # Others:
    "IsolationForest": [
        "_max_features",
        "_max_samples",
        "_decision_path_lengths",
        "_average_path_length_per_tree",
    ],
    "OneClassSVM": [
        "_sparse",
        "_n_support",
        "_probA",
        "_probB",
        "_gamma",
        "_dual_coef_",
        "_intercept_",
        "_effective_probability",
    ],
    "NearestNeighbors": ["_fit_method", "_tree", "_fit_X"],
    "LocalOutlierFactor": [
        "_fit_method",
        "_tree",
        "_fit_X",
        "_distances_fit_X_",
        "_lrd",
    ],
    "TfidfVectorizer": ["_tfidf"],
}

# Private attributes saved for every estimator that has them as plain instance attributes (a
# property, as PCA's `_n_features_out` is, is computed and can't be set on load).
# `_n_features_out` is how ClassNamePrefixFeaturesOutMixin estimators (KMeans, Nystroem, random
# projections, ...) tell in get_feature_names_out that they're fitted.
GENERIC_PRIVATE_ATTRIBUTES: List[str] = ["_n_features_out"]


def _restore_tuple_param(estimator_cls: type, name: str, value: Any) -> Any:
    """
    Turn a param that came back as a list into the tuple its estimator requires. Every format
    stores tuples as lists (JSON has no tuple type, and the type map recording the tuple goes
    through JSON too), but scikit-learn's param validation rejects a list for params declared
    tuple-only - e.g. `MinMaxScaler.feature_range` or `CountVectorizer.ngram_range` - so the
    loaded model couldn't be fitted again. A param is converted only when its class's
    `_parameter_constraints` allow `tuple` but not `list` or "array-like"; classes without
    constraints (most third-party estimators) are left unchanged, and so are params declared
    "no_validation" (a string, not a list).
    """
    if not isinstance(value, list):
        return value
    constraints = getattr(estimator_cls, "_parameter_constraints", {}).get(name, [])
    if (
        isinstance(constraints, list)
        and tuple in constraints
        and list not in constraints
        and "array-like" not in constraints
    ):
        return tuple(value)
    return value


class SklearnSerializer(
    ModelSerializer,
    NumpySerializerMixin,
    ScipySerializerMixin,
):
    """
    Serializer for scikit-learn estimators.

    This class provides methods to convert scikit-learn estimators to and from
    dictionary representations, which can then be used with various format converters.

    The serializer supports a wide range of scikit-learn estimators and handles
    the conversion of numpy arrays and other non-JSON-serializable types.

    Parameters
    ----------
    custom_estimators : callable, list, tuple, or dict, optional
        Optional collection of third-party or custom estimator classes to support during
        serialization and deserialization. This can be:

        - A callable returning an iterable or dict of (name, class) pairs (e.g., a function like ``all_estimators``).
        - A list or tuple of (name, class) pairs.
        - A dict mapping estimator names to their classes.
        - A single (name, class) pair, or a list mixing any of the above (e.g.
          ``[("MyEstimator", MyEstimator), all_estimators]``).

        These estimators are merged into the serializer's internal registry for this instance only,
        allowing support for custom or external estimators without affecting the global registry.

        Classes are identified by (top-level package, class name), so a custom class sharing a
        built-in's name (e.g. chemotools' ``MinMaxScaler``) coexists with it. Only files
        without ``estimator_package`` (format v1-v3) resolve by bare name, where the custom
        class wins.

    trusted_function_modules : iterable of str, optional
        Modules (and their submodules) from which deserialization may import and return
        functions referenced by the file, e.g. the ``func`` of a ``FunctionTransformer``.
        By default only already-imported functions from numpy, scipy, scikit-learn and the
        packages of registered custom estimators are allowed; anything else raises
        ``DeserializationError``. Functions defined in ``__main__`` are never allowed.
        Trusting ``"builtins"`` (e.g. for ``func=abs``) makes every builtin loadable,
        including ``eval``, ``exec`` and ``open``.

    References
    ----------
    .. [1] scikit-learn developer guide:
       https://scikit-learn.org/stable/developers/develop.html

    .. [2] ``sklearn.utils.discovery.all_estimators``:
       https://scikit-learn.org/stable/modules/generated/sklearn.utils.discovery.all_estimators.html

    .. [3] ``skltemplate.utils.discovery.all_estimators`` (project template):
       https://contrib.scikit-learn.org/project-template/generated/skltemplate.utils.discovery.all_estimators.html

    Notes
    -----
    For third-party packages compatible with scikit-learn, it is recommended to implement
    an ``all_estimators()`` utility following the scikit-learn API and template above.
    This enables automatic discovery and integration of custom estimators for serialization.

    If you are maintaining a scikit-learn compatible package, let us know!
    We are happy to extend our testing to include your estimators, ensuring everything works
    smoothly and that we cover any unique types or patterns used in your library.

    To request official support for your package, please open an issue at:
    https://github.com/Gnpd/openmodels/issues

    """

    def __init__(
        self,
        custom_estimators: Optional[
            Union[
                Callable[..., Any],
                List[Any],
                Tuple[Any, ...],
                Dict[str, Type[BaseEstimator]],
            ]
        ] = None,
        trusted_function_modules: Iterable[str] = (),
    ):
        custom_pairs = (
            list(iter_custom_estimators(custom_estimators)) if custom_estimators else []
        )
        # Wrapped in a list: load_custom_estimators treats each list element as one source.
        extra = (
            load_custom_estimators([custom_pairs], ALL_ESTIMATORS)
            if custom_pairs
            else {}
        )
        # Bare-name index, custom classes winning: only for files without estimator_package
        # (format v1-v3). _all_estimators is kept as an alias for anything reading it.
        self._by_name: Dict[str, Type] = {**ALL_ESTIMATORS, **extra}
        self._all_estimators = self._by_name
        # (package, class name) index, built from the raw pairs (not `extra`, which merges
        # same-named classes from different packages and keys by the caller-supplied name).
        self._by_id: Dict[Tuple[str, str], Type] = {
            (_package_of(cls), cls.__name__): cls for cls in ALL_ESTIMATORS.values()
        }
        for _, cls in custom_pairs:
            key = (_package_of(cls), cls.__name__)
            if key in self._by_id and self._by_id[key] is not cls:
                warnings.warn(
                    f"Estimator '{key[0]}.{key[1]}' is registered by two different classes; "
                    f"preferring the later one.",
                    UserWarning,
                )
            self._by_id[key] = cls
        self._custom_packages: Set[str] = {
            _package_of(cls) for _, cls in custom_pairs
        } - {"__main__"}
        self._trusted_function_modules: Tuple[str, ...] = tuple(
            trusted_function_modules
        )
        # Bare names already warned about as ambiguous during the current deserialize() call.
        self._ambiguity_warned: Set[str] = set()
        # Scratch state for one deserialize() call: (node, "prev_leaf_"|"next_leaf_") pairs
        # a Birch _CFNode's leaf-chain pointer couldn't resolve within its own subtree (root_
        # and dummy_leaf_ are independently-deserialized top-level attributes; the pointer
        # crossing between them is fixed up once both are set, see _resolve_birch_leaf_links).
        self._birch_pending_leaf_links: List[Tuple[Any, str]] = []

    # --- Helpers ---
    def _allowed_function_roots(self) -> Set[str]:
        # Registered custom estimators' packages are already imported, so their functions
        # (e.g. a package's own score functions) resolve without importing anything.
        return super()._allowed_function_roots() | self._custom_packages

    def _is_trusted_function_module(self, module_name: str) -> bool:
        return any(
            module_name == trusted or module_name.startswith(trusted + ".")
            for trusted in self._trusted_function_modules
        )

    def _resolve_class(self, data: Dict[str, Any]) -> Type:
        """
        Resolve the class of a serialized estimator node by (package, class name), or by bare
        name for files without "estimator_package" (format v1-v3, custom estimators winning).
        Only registered classes are reachable: nothing named by the file is ever imported.
        """
        name = data["estimator_class"]
        package = data.get("estimator_package")
        if package is not None:
            cls = self._by_id.get((package, name))
            if cls is None:
                raise UnsupportedEstimatorError(
                    f"{package}.{name} is not registered; install '{package}' and pass it "
                    f"via custom_estimators"
                )
            return cls

        cls = self._by_name.get(name)
        if cls is None:
            raise UnsupportedEstimatorError(f"Unknown estimator class '{name}'")
        if name not in self._ambiguity_warned and (
            sum(1 for _, other in self._by_id if other == name) > 1
        ):
            self._ambiguity_warned.add(name)
            warnings.warn(
                f"'{name}' resolved to {_package_of(cls)} (custom estimators win for files "
                f"without estimator_package); re-save the model to record its package",
                UserWarning,
            )
        return cls

    def _check_version(self, stored_version: Optional[str]) -> None:
        """
        Check compatibility between stored scikit-learn version and the current environment.

        Parameters
        ----------
        stored_version : str
            The scikit-learn version recorded during serialization.

        Notes
        -----
        - Issues a warning if the stored version string does not exactly match the current
          version (patch-level differences included).
        - Mentions the versions openmodels has been tested under (TESTED_VERSIONS).
        - Does nothing if no version is stored (for backward compatibility), or if it is
          "unknown" (a placeholder a non-openmodels writer may use).
        """
        if not stored_version or stored_version == "unknown":
            return  # No version info available

        current_version = sklearn.__version__
        if stored_version != current_version:
            warnings.warn(
                f"Version mismatch detected in sklearn deserialization:\n"
                f"- Model serialized with scikit-learn {stored_version}\n"
                f"- Current environment: scikit-learn {current_version}\n\n"
                f"OpenModels has been tested under {TESTED_VERSIONS}. ",
                UserWarning,
            )

    def _check_package_versions(self, packages: Optional[Dict[str, str]]) -> None:
        """
        Check the stored version of every non-scikit-learn package whose estimator classes
        the file needs (the "packages" metadata field) against the installed one.

        Parameters
        ----------
        packages : dict
            ``{package_name: version}`` recorded during serialization.

        Notes
        -----
        - Issues a warning per package whose stored version doesn't exactly match the
          installed one. scikit-learn itself is skipped - `_check_version` covers it.
        - Skips entries whose stored or installed version is "unknown" (nothing to compare).
        - Does nothing if no packages are stored (files predating this field).
        """
        if not packages:
            return

        for name, stored_version in packages.items():
            if name == "sklearn" or stored_version == "unknown":
                continue
            current_version = self._resolve_package_version(name)
            if current_version != "unknown" and stored_version != current_version:
                warnings.warn(
                    f"Version mismatch detected for package '{name}':\n"
                    f"- Model serialized with {name} {stored_version}\n"
                    f"- Current environment: {name} {current_version}",
                    UserWarning,
                )

    def _check_format_version(self, stored_version: Optional[int]) -> None:
        """
        Check compatibility between the stored openmodels wire-format version and the version
        this installed openmodels understands.

        Parameters
        ----------
        stored_version : int
            The openmodels wire-format version recorded during serialization.

        Notes
        -----
        - Issues a warning if the stored version is newer than what this installation
          understands (the file may use a structure introduced after this release).
        - Does nothing if no version is stored - true for every file written before this field
          existed, which all have the version-1 (flat, unversioned) shape.
        """
        if stored_version is None:
            return  # No format version info available - pre-versioning file.

        if stored_version > OPENMODELS_FORMAT_VERSION:
            warnings.warn(
                f"This file was serialized with openmodels wire-format version "
                f"{stored_version}, newer than the format version this installed openmodels "
                f"understands ({OPENMODELS_FORMAT_VERSION}). Deserialization may fail or "
                f"produce incorrect results - consider upgrading openmodels.",
                UserWarning,
            )

    @staticmethod
    def all_estimators(
        type_filter: Optional[str] = None,
    ) -> List[Tuple[str, Type[BaseEstimator]]]:
        """
        Get all scikit-learn supported estimators.

        Parameters
        ----------
        type_filter : str, optional
            If provided, filter estimators by type (e.g., 'classifier', 'regressor').

        Returns
        -------
        list of tuple
            List of (name, class) pairs for supported estimators.
        """

        return [
            (name, cls)
            for name, cls in all_estimators(type_filter=type_filter)
            if name not in NOT_SUPPORTED_ESTIMATORS
        ]

    def _get_nested_types(self, item: Any, in_dict: bool = False) -> Any:
        """
        Recursively determine the type of elements within nested lists and dicts.

        Parameters
        ----------
        item : Any
            The item to inspect for nested types.
        in_dict : bool
            Whether `item` sits inside a typed dict, where a tuple is typed {"tuple": [...]}
            so it comes back as a tuple. Elsewhere a tuple is typed as a tuple of its
            elements' types, which formats save as a list, as before v4.

        Returns
        -------
        Any
            A nested list representing the types of elements in the input item.

        Examples
        ---------

        [1, [1, 2, [1, 2, 3]], 2] -> ['int',['int','int','ndarray'],'int']
        {"a": np.zeros(2), "b": (1, 2)} -> {"dict": {"a": "ndarray", "b": {"tuple": ["int", "int"]}}}
        {"a": 1, "b": [1.0, 2.0]} -> "dict"

        """
        # Before the ndarray checks: a masked array is an ndarray subclass.
        if isinstance(item, np.ma.MaskedArray):
            return "MaskedArray"

        if isinstance(item, dict):
            return self._get_dict_types(item)

        # Handle np.ndarray of estimators
        if (
            isinstance(item, np.ndarray)
            and item.dtype == np.dtype("O")
            and item.size > 0
            and isinstance(item.ravel()[0], BaseEstimator)
        ):
            return "estimators_collection"

        # Handle lists or tuple of estimators
        if (
            isinstance(item, (list, tuple))
            and item
            and isinstance(item[0], BaseEstimator)
        ):
            return "estimators_collection"

        # Handle tuples explicitly
        if isinstance(item, tuple):
            if in_dict:
                return {"tuple": [self._get_nested_types(s, True) for s in item]}
            return tuple(self._get_nested_types(subitem) for subitem in item)

        # Handle lists
        if isinstance(item, list):
            return [self._get_nested_types(subitem, in_dict) for subitem in item]

        elif isinstance(item, BaseEstimator):
            # For estimators, return their class name instead of just 'BaseEstimator'
            return item.__class__.__name__
        elif isinstance(item, np.dtype):
            # Normalize to a single stable tag: concrete np.dtype instances are actually
            # instances of numpy-internal per-dtype subclasses (Float64DType, Int64DType,
            # BoolDType, ...), so type(item).__name__ is not a stable/registrable tag.
            return "dtype"
        elif issparse(item):
            # Normalize every scipy sparse container (csr_matrix, csc_matrix, csr_array,
            # csc_array, ...) to the one tag ScipySerializerMixin actually registers a
            # deserializer for - _serialize_csr_matrix already converts any of them to csr_matrix
            # via the csr_matrix(value) constructor, so type(item).__name__ (e.g. "csr_array")
            # would tag a value the deserializer dispatch table has no matching entry for.
            return "csr_matrix"
        else:
            # Return the type name if it's not a list or it's an empty list
            return type(item).__name__

    def _get_dict_types(self, item: dict) -> Any:
        """
        Type of a dict: {"dict": {key: type}} when a value needs restoring on load, plus
        {"key_types": {key: type}} for its non-string keys, which are saved as text (see
        `_key_to_text`); "dict" when every key is a string and every value plain JSON. A Bunch
        is always typed {"Bunch": {...}}, so it comes back as a Bunch.
        """
        value_types = {
            _key_to_text(key): self._get_nested_types(value, in_dict=True)
            for key, value in item.items()
        }
        key_types = {
            _key_to_text(key): self._get_nested_types(key)
            for key in item
            if not isinstance(key, str)
        }
        if isinstance(item, Bunch):
            return {"Bunch": value_types}
        if key_types:
            return {"dict": value_types, "key_types": key_types}
        if all(_is_plain_type(t) for t in value_types.values()):
            return "dict"
        return {"dict": value_types}

    def _get_nested_dtypes(self, item: Any) -> Any:
        """dtypes for the values of a typed dict, mirroring it: an array's dtype, a dict of
        them for a nested dict, a list of them for a list. None when nothing inside needs one
        (a masked array records its own)."""
        if isinstance(item, np.ma.MaskedArray):
            return None
        if isinstance(item, np.ndarray):
            return str(item.dtype)
        if isinstance(item, dict):
            dtypes = {
                _key_to_text(key): dtype
                for key, value in item.items()
                if (dtype := self._get_nested_dtypes(value)) is not None
            }
            return dtypes or None
        if isinstance(item, (list, tuple)):
            dtypes_list = [self._get_nested_dtypes(value) for value in item]
            return dtypes_list if any(d is not None for d in dtypes_list) else None
        return None

    def _get_type_maps(self, values_dict: dict) -> tuple[dict, dict]:
        """
        Given a dict of raw values (e.g. model attributes or params),
        builds the corresponding types/dtypes maps.
        """
        types_map = {
            key: self._get_nested_types(value) for key, value in values_dict.items()
        }
        dtypes_map = {
            key: self._get_dtype(value)
            for key, value in values_dict.items()
            if isinstance(value, np.ndarray)
            or (isinstance(value, (list, tuple)) and value)  # non-empty list/tuple
        }

        # Ensure tuples are not included in dtypes_map
        for key, value in values_dict.items():
            if isinstance(value, tuple):
                dtypes_map.pop(key, None)  # Remove tuples from dtypes_map

        # A typed dict's dtypes mirror its values (see _get_nested_dtypes).
        for key, value in values_dict.items():
            if isinstance(types_map[key], dict):
                dict_dtypes = self._get_nested_dtypes(value)
                if dict_dtypes is not None:
                    dtypes_map[key] = dict_dtypes

        return types_map, dtypes_map

    def _extract_estimator_attributes(self, estimator: BaseEstimator) -> Dict[str, Any]:
        """
        Extract fitted sklearn attributes,
        """

        def is_valid_attribute(key: str) -> bool:
            return (
                not key.startswith("__")  # not private/internal
                and not key.startswith("_")  # not protected
                and key.endswith("_")  # sklearn convention
                and not key.endswith("__")  # not dunder
                and not isinstance(
                    getattr(type(estimator), key, None), property
                )  # not property
                and not callable(getattr(estimator, key))  # not method
            )

        # Collect attributes
        attribute_keys = [key for key in dir(estimator) if is_valid_attribute(key)]
        attribute_keys += ATTRIBUTE_EXCEPTIONS.get(estimator.__class__.__name__, [])
        attribute_keys += [
            key for key in GENERIC_PRIVATE_ATTRIBUTES if key in vars(estimator)
        ]

        # Prevents attribute exceptions introduced in newer scikit-learn versions from breaking older versions of the serializer
        attributes = {
            key: getattr(estimator, key)
            for key in attribute_keys
            if hasattr(estimator, key)
        }

        # A neighbors estimator's search tree is rebuilt on load from its own state (see
        # _rebuild_neighbors_tree), so a stored tree is never used. A KDTree is still written
        # only so openmodels 0.2.2 can read the file; no older reader can load a BallTree, so
        # it's left out (smaller files).
        if isinstance(estimator, NeighborsBase) and isinstance(
            attributes.get("_tree"), BallTree
        ):
            del attributes["_tree"]

        # A search's scorer_ is rebuilt on load from its `scoring` param (see
        # _rebuild_derived_attributes). Scorer objects can't be written: a multi-metric
        # scorer_ (a dict of them) made saving fail.
        if isinstance(estimator, BaseSearchCV):
            attributes.pop("scorer_", None)

        return attributes

    def convert_from_serializable(
        self, value: Any, value_type: Any = "none", value_dtype: Optional[str] = None
    ) -> Any:
        # Every estimator node goes through _deserialize_core (and so _resolve_class),
        # whatever its type tag: an unregistered class name has no tag handler and would
        # otherwise silently come back as a raw dict instead of raising.
        if (
            isinstance(value, dict)
            and "estimator_class" in value
            and isinstance(value_type, str)
            and value_type != "dict"
        ):
            return self._deserialize_core(value)
        # Same for Gaussian-process kernels ({"kernel_type", "params"}): routing by content
        # covers every kernel class, and an unknown kernel raises instead of loading as a dict.
        if (
            isinstance(value, dict)
            and "kernel_type" in value
            and "params" in value
            and isinstance(value_type, str)
            and value_type != "dict"
        ):
            return self._deserialize_kernel(value)
        # A Bunch (e.g. Voting*/Stacking*'s named_estimators_) is typed like a dict, under
        # its own name (see _get_dict_types).
        if (
            isinstance(value_type, dict)
            and len(value_type) == 1
            and isinstance(value_type.get("Bunch"), dict)
            and isinstance(value, dict)
        ):
            return Bunch(
                **self._deserialize_typed_dict(value, value_type["Bunch"], value_dtype)
            )
        return super().convert_from_serializable(value, value_type, value_dtype)

    # --- Handlers ---
    def _get_serializer_handlers(self):
        # important to run before super() to deal with possible np.ndarray of estimators
        return [
            (BaseEstimator, self._serialize_core),
            (BaseLoss, self._serialize_loss),
            ((KDTree, BallTree), self._serialize_search_tree),
            (Kernel, self._serialize_kernel),
            (Tree, self._serialize_tree),
            (TreePredictor, self._serialize_tree_predictor),
            # Before np.ndarray: a masked array is an ndarray subclass.
            (np.ma.MaskedArray, self._serialize_masked_array),
            (np.ndarray, self._serialize_estimators_collection),
            (_CalibratedClassifier, self._serialize_calibrated_classifier),
            (_BisectingTree, self._serialize_bisecting_tree),
            (_CurveScorer, self._serialize_curve_scorer),
            (_CFNode, self._serialize_cfnode),
            # Callable, so it must come before the generic function handler in super().
            (make_column_selector, self._serialize_column_selector),
        ] + super()._get_serializer_handlers()

    def _get_deserializer_handlers(self):
        # Register losses
        loss_handlers = [
            (loss_name, (lambda v, ln=loss_name: self._deserialize_loss(v, ln)))
            for loss_name in LOSS_CLASS_REGISTRY.keys()
        ]
        # Estimators
        estimator_handlers = [
            (est_name, self._deserialize_core) for est_name in self._by_name.keys()
        ]
        return (
            [
                ("estimators_collection", self._deserialize_estimators_collection),
                ("TreePredictor", self._deserialize_tree_predictor),
                ("_BisectingTree", self._deserialize_bisecting_tree),
                ("_CalibratedClassifier", self._deserialize_calibrated_classifier),
                ("_CurveScorer", self._deserialize_curve_scorer),
                ("_CFNode", self._deserialize_cfnode),
                ("make_column_selector", self._deserialize_column_selector),
                ("MaskedArray", self._deserialize_masked_array),
            ]
            + loss_handlers
            + estimator_handlers
            + super()._get_deserializer_handlers()
        )

    # --- Sklearn specific serializers/deserializers ---
    def _serialize_bisecting_tree(self, tree: _BisectingTree) -> dict:
        # center/indices are serialized with an explicit dtype (rather than relying on the
        # generic convert_to_serializable, which loses dtype via a plain .tolist()) so that
        # e.g. a float32-fitted tree doesn't get silently widened to JSON's float64 on the
        # way back - predict()'s Cython inner loop requires the exact original buffer dtype.
        center = np.asarray(tree.center)
        indices = np.asarray(tree.indices)
        return {
            "center": self.convert_to_serializable(center),
            "center_dtype": str(center.dtype),
            "indices": self.convert_to_serializable(indices),
            "indices_dtype": str(indices.dtype),
            "score": tree.score,
            "label": getattr(tree, "label", None),
            "left": self._serialize_bisecting_tree(tree.left) if tree.left else None,
            "right": self._serialize_bisecting_tree(tree.right) if tree.right else None,
        }

    def _deserialize_bisecting_tree(self, data: dict) -> _BisectingTree:
        if data is None:
            return None
        node = _BisectingTree(
            center=np.array(data["center"], dtype=data.get("center_dtype")),
            indices=np.array(data["indices"], dtype=data.get("indices_dtype")),
            score=data["score"],
        )
        if data.get("label") is not None:
            node.label = data["label"]
        node.left = self._deserialize_bisecting_tree(data["left"])
        node.right = self._deserialize_bisecting_tree(data["right"])
        return node

    def _serialize_masked_array(self, value: np.ma.MaskedArray) -> Dict[str, Any]:
        """
        A masked array (a search's cv_results_["param_*"] columns): data, mask, shape and
        dtype. Masked slots are written as the array's fill value, or None for object arrays,
        whose elements (strings, None, dicts, estimators, ...) are typed one by one.
        """
        mask = np.ma.getmaskarray(value).ravel()
        serialized: Dict[str, Any] = {
            "mask": mask.tolist(),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        if value.dtype == np.dtype("O"):
            items = [
                None if masked else item
                for item, masked in zip(value.data.ravel().tolist(), mask)
            ]
            serialized["data"] = self.convert_to_serializable(items)
            serialized["types"] = [
                self._get_nested_types(item, in_dict=True) for item in items
            ]
        else:
            serialized["data"] = self.convert_to_serializable(value.filled().ravel())
        return serialized

    def _deserialize_masked_array(self, data: Dict[str, Any]) -> np.ma.MaskedArray:
        shape = tuple(data["shape"])
        mask = np.array(data["mask"], dtype=bool).reshape(shape)
        if "types" in data:
            items = [
                self.convert_from_serializable(item, item_type)
                for item, item_type in zip(data["data"], data["types"])
            ]
            # Filled one by one: np.array would turn list or tuple elements into dimensions.
            values = np.empty(len(items), dtype=object)
            for i, item in enumerate(items):
                values[i] = item
        else:
            values = np.array(data["data"], dtype=np.dtype(data["dtype"]))
        return np.ma.MaskedArray(values.reshape(shape), mask=mask)

    def _serialize_calibrated_classifier(
        self, obj: _CalibratedClassifier
    ) -> Dict[str, Any]:
        # Serialize estimator, calibrators (list), classes, and method
        return {
            "estimator": self.convert_to_serializable(obj.estimator),
            "calibrators": self.convert_to_serializable(obj.calibrators),
            "classes": self.convert_to_serializable(obj.classes),
            "method": obj.method,
        }

    def _deserialize_calibrated_classifier(
        self, data: Dict[str, Any]
    ) -> _CalibratedClassifier:
        estimator = self._deserialize_core(data["estimator"])
        calibrators = [self._deserialize_core(c) for c in data["calibrators"]]
        classes = np.array(data["classes"])
        method = data["method"]
        return _CalibratedClassifier(
            estimator, calibrators, classes=classes, method=method
        )

    def _serialize_cfnode(self, node: _CFNode) -> Dict[str, Any]:
        """
        Recursively serialize a Birch _CFNode and its full subtree of _CFSubcluster/_CFNode
        descendants (a _CFSubcluster's `child_` is where the tree actually recurses one level
        down). Captures the real fitted state - not just the node's scalar config - so
        predict()/partial_fit() keep working after a round-trip.

        Also captures the doubly-linked prev_leaf_/next_leaf_ chain Birch threads across leaf
        nodes only (used by Birch._get_leaves() for fast traversal). A leaf's neighbor can live
        outside this node's own subtree - e.g. root_'s leftmost leaf's prev_leaf_ is
        Birch.dummy_leaf_, a sibling top-level attribute serialized independently - such refs
        are tagged "external" and cross-linked after deserialization by
        _resolve_birch_leaf_links, since node_id namespaces are local to each top-level
        _serialize_cfnode call.
        """
        node_ids: Dict[int, int] = {}

        def get_id(n: _CFNode) -> int:
            key = id(n)
            if key not in node_ids:
                node_ids[key] = len(node_ids)
            return node_ids[key]

        # Assign every reachable node an id up front so leaf_ref can resolve a ref to a node
        # not yet visited by the (depth-first) serialize_node walk below.
        def collect(n: _CFNode) -> None:
            get_id(n)
            for sub in n.subclusters_:
                if sub.child_ is not None:
                    collect(sub.child_)

        collect(node)

        def leaf_ref(neighbor: Optional[_CFNode]) -> Union[int, str, None]:
            if neighbor is None:
                return None
            return node_ids.get(id(neighbor), "external")

        def serialize_subcluster(sub: _CFSubcluster) -> Dict[str, Any]:
            centroid = np.asarray(sub.centroid_)
            linear_sum = np.asarray(sub.linear_sum_)
            return {
                "n_samples_": sub.n_samples_,
                # squared_sum_/sq_norm_ are np.dot(...) results - numpy scalars (e.g.
                # np.float32), not plain Python floats - so they need convert_to_serializable's
                # np.generic handling (.item()) to be JSON-safe.
                "squared_sum_": self.convert_to_serializable(sub.squared_sum_),
                "sq_norm_": self.convert_to_serializable(sub.sq_norm_),
                "linear_sum_": self.convert_to_serializable(linear_sum),
                "linear_sum_dtype": str(linear_sum.dtype),
                "centroid_": self.convert_to_serializable(centroid),
                "centroid_dtype": str(centroid.dtype),
                "child": serialize_node(sub.child_) if sub.child_ is not None else None,
            }

        def serialize_node(n: _CFNode) -> Dict[str, Any]:
            return {
                "node_id": get_id(n),
                "threshold": n.threshold,
                "branching_factor": n.branching_factor,
                "is_leaf": n.is_leaf,
                "n_features": n.n_features,
                "dtype": str(n.init_centroids_.dtype),
                "subclusters": [serialize_subcluster(sub) for sub in n.subclusters_],
                "prev_leaf_ref": leaf_ref(n.prev_leaf_) if n.is_leaf else None,
                "next_leaf_ref": leaf_ref(n.next_leaf_) if n.is_leaf else None,
            }

        return serialize_node(node)

    def _deserialize_cfnode(self, data: dict) -> Optional[_CFNode]:
        """
        Deserialize a Birch _CFNode subtree (inverse of _serialize_cfnode). Rebuilds each node
        by replaying the real _CFNode.append_subcluster()/_CFSubcluster construction path
        rather than hand-assembling the init_centroids_/centroids_ view relationship, for
        structural parity with what Birch.fit() itself produces.

        Old (pre-fix) serialized files only ever captured a node's scalar config (no
        "subclusters"/"node_id"/leaf-ref keys) - those fields are read via .get() with the
        same defaults the old stub used, so old files keep deserializing to the same
        structure-less (broken-but-non-crashing) node as before, not retroactively fixed.
        """
        if data is None:
            return None

        nodes: Dict[int, _CFNode] = {}

        def deserialize_subcluster(sub_data: dict) -> _CFSubcluster:
            linear_sum = np.array(
                self.convert_from_serializable(sub_data["linear_sum_"]),
                dtype=sub_data.get("linear_sum_dtype"),
            )
            subcluster = _CFSubcluster(linear_sum=linear_sum)
            subcluster.n_samples_ = sub_data["n_samples_"]
            subcluster.squared_sum_ = sub_data["squared_sum_"]
            subcluster.sq_norm_ = sub_data["sq_norm_"]
            subcluster.centroid_ = np.array(
                self.convert_from_serializable(sub_data["centroid_"]),
                dtype=sub_data.get("centroid_dtype"),
            )
            if sub_data.get("child") is not None:
                subcluster.child_ = deserialize_node(sub_data["child"])
            return subcluster

        def deserialize_node(node_data: dict) -> _CFNode:
            node = _CFNode(
                threshold=node_data["threshold"],
                branching_factor=node_data["branching_factor"],
                is_leaf=node_data["is_leaf"],
                n_features=node_data["n_features"],
                dtype=np.dtype(node_data.get("dtype") or np.float64),
            )
            node_id = node_data.get("node_id")
            if node_id is not None:
                nodes[node_id] = node
            for sub_data in node_data.get("subclusters", []):
                node.append_subcluster(deserialize_subcluster(sub_data))
            return node

        root = deserialize_node(data)

        # Second pass: wire up leaf refs now that every node in this subtree has been built
        # (and thus has an id in `nodes`, regardless of forward/backward reference order).
        def link_leaves(node_data: dict) -> None:
            if node_data["is_leaf"]:
                node = nodes[node_data["node_id"]]
                for ref_key, attr in (
                    ("prev_leaf_ref", "prev_leaf_"),
                    ("next_leaf_ref", "next_leaf_"),
                ):
                    ref = node_data.get(ref_key)
                    if ref is None:
                        setattr(node, attr, None)
                    elif ref == "external":
                        self._birch_pending_leaf_links.append((node, attr))
                    else:
                        setattr(node, attr, nodes[ref])
            for sub_data in node_data.get("subclusters", []):
                if sub_data.get("child") is not None:
                    link_leaves(sub_data["child"])

        if "node_id" in data:
            link_leaves(data)

        return root

    def _resolve_birch_leaf_links(self) -> None:
        """
        Cross-link the two independently-deserialized Birch._CFNode graphs (root_ and
        dummy_leaf_): dummy_leaf_.next_leaf_ always points at the current globally-leftmost
        leaf inside root_'s tree, and that leaf's prev_leaf_ points back at dummy_leaf_. Since
        root_ and dummy_leaf_ are deserialized as independent top-level attributes (in
        whichever order they appear in the serialized data), _deserialize_cfnode can't resolve
        this pointer itself - it queues each side on self._birch_pending_leaf_links, and this
        is called once both attributes are guaranteed to be set.
        """
        pending = self._birch_pending_leaf_links
        self._birch_pending_leaf_links = []
        prev_pending = [node for node, attr in pending if attr == "prev_leaf_"]
        next_pending = [node for node, attr in pending if attr == "next_leaf_"]
        if len(prev_pending) == 1 and len(next_pending) == 1:
            prev_pending[0].prev_leaf_ = next_pending[0]
            next_pending[0].next_leaf_ = prev_pending[0]
        elif pending:
            warnings.warn(
                "Could not resolve Birch's dummy_leaf_/root_ leaf-chain link after "
                "deserialization (unexpected pending link shape); partial_fit()/predict() may "
                "not walk the full leaf chain.",
                UserWarning,
            )

    def _serialize_tree(self, tree: Tree) -> Dict[str, Any]:
        """
        Serializes a sklearn.tree._tree.Tree object to a dictionary.

        Parameters:
            tree (sklearn.tree._tree.Tree): The internal tree structure from a fitted tree-based model (e.g., model.tree_)

        Returns:
            dict: Serialized tree attributes
        """
        state = tree.__getstate__()

        return {
            "n_features": tree.n_features,
            "n_outputs": tree.n_outputs,
            "n_classes": tree.n_classes.tolist(),
            "state": {
                k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in state.items()
            },
            "nodes_dtype": [list(t) for t in state["nodes"].dtype.descr],  # for JSON
        }

    def _deserialize_tree(self, tree_data: Dict[str, Any]) -> Tree:
        """
        Deserializes a dictionary representation of a tree back to a sklearn.tree._tree.Tree object.

        """
        tree = Tree(
            tree_data["n_features"],
            np.array(tree_data["n_classes"], dtype=np.intp),
            tree_data["n_outputs"],
        )

        state = {}
        for key, value in tree_data["state"].items():
            if key == "nodes":
                # Restore dtype
                nodes_dtype_descr = [
                    tuple(field) for field in tree_data["nodes_dtype"] if field[0] != ""
                ]
                nodes_dtype = np.dtype(nodes_dtype_descr)
                if isinstance(value, list) and isinstance(value[0], list):
                    value = [tuple(row) for row in value]
                state["nodes"] = np.array(value, dtype=nodes_dtype)
            else:
                state[key] = np.array(value)

        tree.__setstate__(state)
        return tree

    def _serialize_tree_predictor(self, predictor: TreePredictor) -> Dict[str, Any]:
        """
        Serialize a sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor object.
        """
        return {
            "nodes": self.convert_to_serializable(predictor.nodes),
            "binned_left_cat_bitsets": self.convert_to_serializable(
                predictor.binned_left_cat_bitsets
            ),
            "raw_left_cat_bitsets": self.convert_to_serializable(
                predictor.raw_left_cat_bitsets
            ),
        }

    def _deserialize_tree_predictor(self, data: Dict[str, Any]) -> TreePredictor:
        node_dtype = np.dtype(
            [
                ("value", "<f8"),
                ("count", "<u4"),
                ("feature_idx", "<i8"),
                ("num_threshold", "<f8"),
                ("missing_go_to_left", "u1"),
                ("left", "<u4"),
                ("right", "<u4"),
                ("gain", "<f8"),
                ("depth", "<u4"),
                ("is_leaf", "u1"),
                ("bin_threshold", "u1"),
                ("is_categorical", "u1"),
                ("bitset_idx", "<u4"),
            ]
        )
        nodes_list = [tuple(row) for row in data["nodes"]]
        nodes = np.array(nodes_list, dtype=node_dtype)

        def ensure_2d_uint32(arr):
            arr = np.array(arr, dtype="uint32")
            if arr.ndim == 1:
                # If empty, shape should be (0, 8)
                if arr.size == 0:
                    arr = arr.reshape((0, 8))
                else:
                    arr = arr.reshape((-1, 8))
            return arr

        binned_left_cat_bitsets = ensure_2d_uint32(data["binned_left_cat_bitsets"])
        raw_left_cat_bitsets = ensure_2d_uint32(data["raw_left_cat_bitsets"])

        return TreePredictor(
            nodes=nodes,
            binned_left_cat_bitsets=binned_left_cat_bitsets,
            raw_left_cat_bitsets=raw_left_cat_bitsets,
        )

    def _serialize_loss(self, value: BaseLoss) -> Dict[str, Any]:
        """
        Serialize a scikit-learn loss object using its constructor parameters.

        Parameters:
            obj: The loss object instance.

        Returns:
            dict: Serialized representation.
        """
        cls = type(value)
        params = {
            k: getattr(value, k, None)
            for k in inspect.signature(cls.__init__).parameters
            if k != "self"
        }
        # Fix for TweedieRegressor: ensure 'power' is not None
        if "power" in params and params["power"] is None:
            params["power"] = getattr(value, "power", 0.0)
        return {"params": params}

    def _deserialize_loss(self, value: Dict[str, Any], loss_name: str) -> BaseLoss:
        loss_cls = LOSS_CLASS_REGISTRY[loss_name]
        params = value.get("params", {})
        return loss_cls(**params)

    def _serialize_search_tree(self, value: Union[KDTree, BallTree]) -> Dict[str, Any]:
        """
        Serialize a KDTree/BallTree as the inputs it's rebuilt from: its data and its sample
        weights (None when unweighted). The weights are stored only inside the tree - e.g.
        KernelDensity keeps them nowhere else. The metric and leaf_size live on the owning
        estimator, which rebuilds the tree with them on load (`_rebuild_neighbors_tree`,
        `_rebuild_kernel_density_tree`). dtype is captured explicitly (same reasoning as
        _serialize_bisecting_tree) so non-float64 data isn't silently widened by JSON.
        """
        data = np.array(value.data)
        sample_weight = value.sample_weight
        return {
            "data": self.convert_to_serializable(data),
            "data_dtype": str(data.dtype),
            "sample_weight": (
                None
                if sample_weight is None
                else self.convert_to_serializable(np.asarray(sample_weight))
            ),
        }

    def _deserialize_search_tree(
        self, tree_data: Dict[str, Any], tree_cls: Type[Union[KDTree, BallTree]]
    ) -> Union[KDTree, BallTree]:
        """
        Rebuild a KDTree/BallTree from its data and sample weights, with the default metric
        and leaf_size. Estimators that own a tree rebuild it again with their real metric
        (see `_serialize_search_tree`). Files written before 0.2.3 have no "sample_weight".
        """
        data = np.array(tree_data["data"], dtype=tree_data.get("data_dtype"))
        sample_weight = tree_data.get("sample_weight")
        return tree_cls(
            data,
            sample_weight=None if sample_weight is None else np.asarray(sample_weight),
        )

    def _rebuild_kernel_density_tree(self, model: BaseEstimator) -> None:
        """
        Rebuild a KernelDensity's `tree_` exactly as `KernelDensity.fit` builds it: the loaded
        tree's class, data and sample weights, plus the estimator's `metric`, `leaf_size` and
        `metric_params` (which the loaded tree, built with defaults, lacks).
        """
        tree = getattr(model, "tree_", None)
        if not isinstance(model, KernelDensity) or not isinstance(
            tree, (KDTree, BallTree)
        ):
            return
        sample_weight = tree.sample_weight
        model.tree_ = type(tree)(
            np.asarray(tree.data),
            metric=model.metric,
            leaf_size=model.leaf_size,
            sample_weight=None if sample_weight is None else np.asarray(sample_weight),
            **(model.metric_params or {}),
        )

    def _rebuild_neighbors_tree(self, model: BaseEstimator) -> None:
        """
        Rebuild a neighbors estimator's search tree (`_tree`) from its own fitted state,
        exactly as `NeighborsBase._fit` builds it: `KDTree`/`BallTree` over `_fit_X` with
        `leaf_size`, `effective_metric_` and `effective_metric_params_`. The tree is
        deterministic in those inputs, so the result is identical to the original.

        Rebuilding instead of restoring a stored tree means files never need to carry one:
        regressors never stored `_tree` at all, and the trees older files did store were
        rebuilt as `KDTree(data)`, dropping the metric and leaf_size (wrong neighbors for any
        non-euclidean metric). Models fitted with `brute` (e.g. on sparse data) have no tree,
        and files missing any of the inputs are left as they are.
        """
        if not isinstance(model, NeighborsBase) or not all(
            hasattr(model, attr) for attr in _NEIGHBORS_TREE_INPUTS
        ):
            return
        fit_method = model._fit_method
        if fit_method not in ("kd_tree", "ball_tree"):
            model._tree = None  # "brute", as NeighborsBase._fit sets it
            return
        tree_cls = KDTree if fit_method == "kd_tree" else BallTree
        model._tree = tree_cls(
            model._fit_X,
            model.leaf_size,
            metric=model.effective_metric_,
            **model.effective_metric_params_,
        )

    def _rebuild_derived_attributes(self, model: BaseEstimator) -> None:
        """
        Recompute private attributes that `fit` derives from fitted state openmodels does save,
        exactly as `fit` computes them, so files without them (every file written by 0.2.x)
        load complete:

        - `HistGradientBoosting*._loss`, needed by `predict_proba` (the regressor's is also
          saved). `sample_weight` only changes the loss during training, never its link.
        - `HistGradientBoosting*._n_features`, needed by `staged_predict*`: the input width.
        - `LinearDiscriminantAnalysis._max_components`, needed by `transform`.
        - A search's (`GridSearchCV`, ...) `scorer_`, needed by `score`: built from its `scoring`
          param by `_get_scorers`, as `fit` does. It's never saved (see
          `_extract_estimator_attributes`).
        """
        if isinstance(
            model, (HistGradientBoostingClassifier, HistGradientBoostingRegressor)
        ) and hasattr(model, "n_trees_per_iteration_"):
            if not hasattr(model, "_loss"):
                model._loss = model._get_loss(sample_weight=None)
            if not hasattr(model, "_n_features"):
                model._n_features = model.n_features_in_
        if (
            isinstance(model, LinearDiscriminantAnalysis)
            and hasattr(model, "classes_")
            and not hasattr(model, "_max_components")
        ):
            max_components = min(len(model.classes_) - 1, model.n_features_in_)
            model._max_components = (
                max_components if model.n_components is None else model.n_components
            )
        if (
            isinstance(model, BaseSearchCV)
            and hasattr(model, "multimetric_")
            and not hasattr(model, "scorer_")
        ):
            scorers, _ = model._get_scorers()
            model.scorer_ = (
                scorers._scorers if isinstance(scorers, _MultimetricScorer) else scorers
            )

    def _serialize_estimators_collection(
        self, value: Union[np.ndarray, List[BaseEstimator]]
    ) -> List[Any]:
        # Accept both numpy arrays and lists of estimators
        if isinstance(value, np.ndarray):
            if (
                value.dtype == np.dtype("O")
                and value.size > 0
                and isinstance(value.ravel()[0], BaseEstimator)
            ):
                return [
                    [self.convert_to_serializable(est) for est in row] for row in value
                ]
            return self._serialize_ndarray(value)

        if (
            isinstance(value, (list, tuple))
            and value
            and isinstance(value[0], BaseEstimator)
        ):
            return [self.convert_to_serializable(est) for est in value]
        return value

    def _deserialize_estimators_collection(
        self, value: List[Any]
    ) -> Union[np.ndarray, List[BaseEstimator]]:
        # Handle list of lists (array) or flat list (meta-estimator)
        if isinstance(value, list) and value:
            if isinstance(value[0], list):
                # 2D array
                arr = []
                for row in value:
                    arr.append(
                        [
                            (
                                self._deserialize_core(est)
                                if isinstance(est, dict) and "estimator_class" in est
                                else est
                            )
                            for est in row
                        ]
                    )
                return np.array(arr, dtype=object)
            else:
                # Flat list
                return [
                    (
                        self._deserialize_core(est)
                        if isinstance(est, dict) and "estimator_class" in est
                        else est
                    )
                    for est in value
                ]
        return value

    def _serialize_kernel(self, kernel: Kernel) -> Dict[str, Any]:
        """
        Recursively serialize a sklearn.gaussian_process.kernels.Kernel object as its class
        name and constructor params. Params that are kernels - also inside lists, e.g.
        CompoundKernel's `kernels` - are serialized recursively; every other value goes through
        convert_to_serializable (e.g. a fitted anisotropic length_scale ndarray).
        """
        return {
            "kernel_type": type(kernel).__name__,
            "params": {
                k: self._serialize_kernel_value(v)
                for k, v in kernel.get_params(deep=False).items()
            },
        }

    def _serialize_kernel_value(self, value: Any) -> Any:
        if isinstance(value, Kernel):
            return self._serialize_kernel(value)
        if isinstance(value, (list, tuple)):
            return [self._serialize_kernel_value(v) for v in value]
        return self.convert_to_serializable(value)

    def _deserialize_kernel(self, data: Dict[str, Any]) -> Kernel:
        """
        Recursively deserialize a kernel dict back to a Kernel object. `kernel_type` comes from
        the file, so it must name a concrete Kernel class in sklearn.gaussian_process.kernels
        (looked up as an attribute of that already-imported module; nothing is imported), never
        any other callable there.
        """
        kernel_type = data.get("kernel_type")
        kernel_cls = (
            getattr(_gp_kernels, kernel_type, None)
            if isinstance(kernel_type, str) and not kernel_type.startswith("_")
            else None
        )
        if not (
            isinstance(kernel_cls, type)
            and issubclass(kernel_cls, Kernel)
            and not inspect.isabstract(kernel_cls)
        ):
            raise DeserializationError(f"Unknown kernel type '{kernel_type}'")
        return kernel_cls(
            **{
                k: self._deserialize_kernel_value(v)
                for k, v in data.get("params", {}).items()
            }
        )

    def _deserialize_kernel_value(self, value: Any) -> Any:
        if isinstance(value, dict) and "kernel_type" in value:
            return self._deserialize_kernel(value)
        if isinstance(value, list):
            return [self._deserialize_kernel_value(v) for v in value]
        return value

    def _serialize_column_selector(
        self, selector: make_column_selector
    ) -> Dict[str, Any]:
        """
        Serialize a ColumnTransformer `make_column_selector` as its three settings. A dtype
        spec is a string ("number"), a type (np.number, float) or a list of these; types are
        written as `_serialize_type` does.
        """

        def dtype_spec(spec: Any) -> Any:
            if isinstance(spec, (list, tuple)):
                return [dtype_spec(s) for s in spec]
            if isinstance(spec, type):
                return self._serialize_type(spec)
            return spec

        return {
            "pattern": selector.pattern,
            "dtype_include": dtype_spec(selector.dtype_include),
            "dtype_exclude": dtype_spec(selector.dtype_exclude),
        }

    def _deserialize_column_selector(
        self, data: Dict[str, Any]
    ) -> make_column_selector:
        """Rebuild a `make_column_selector` from its saved settings. Only this fixed class is
        constructed, and type names go through `_deserialize_type`."""

        def dtype_spec(spec: Any) -> Any:
            if isinstance(spec, list):
                return [dtype_spec(s) for s in spec]
            if isinstance(spec, dict) and "type_name" in spec:
                return self._deserialize_type(spec)
            return spec

        return make_column_selector(
            pattern=data.get("pattern"),
            dtype_include=dtype_spec(data.get("dtype_include")),
            dtype_exclude=dtype_spec(data.get("dtype_exclude")),
        )

    def _serialize_curve_scorer(self, scorer: _CurveScorer) -> Dict[str, Any]:
        # Find the scorer name in sklearn.metrics.get_scorer_names()
        score_func = None
        for name in get_scorer_names():
            try:
                registered = get_scorer(name)
                # Compare function and kwargs
                if (
                    hasattr(registered, "_score_func")
                    and registered._score_func == scorer._score_func
                    and getattr(registered, "_kwargs", {})
                    == getattr(scorer, "_kwargs", {})
                ):
                    score_func = name
                    break
            except Exception:
                continue

        return {
            "score_func": score_func,
            "sign": scorer._sign,
            "kwargs": scorer._kwargs,
            "thresholds": scorer._thresholds,
            "response_method": scorer._response_method,
        }

    def _deserialize_curve_scorer(self, data: Dict[str, Any]) -> _CurveScorer:
        from sklearn.metrics import get_scorer

        score_func_name = data["score_func"]
        if score_func_name is not None:
            # Get the base scorer (e.g. accuracy, f1, etc.)
            base_scorer = get_scorer(score_func_name)
            # Use from_scorer to reconstruct the _CurveScorer
            return _CurveScorer.from_scorer(
                base_scorer,
                response_method=data.get("response_method", "predict"),
                thresholds=data.get("thresholds"),
            )
        else:
            raise ValueError(
                "Cannot deserialize custom/non-standard _CurveScorer functions."
            )

    def _serialize_core(self, model: BaseEstimator) -> Dict[str, Any]:
        """
        Serialize a scikit-learn estimator to a dictionary, without the root-only
        "metadata" block (see `serialize`). This is the method used for every
        recursive/nested estimator (a `Pipeline` step, a `VotingClassifier`'s
        `estimators_`, ...) so that "metadata" is never duplicated below the true root.

        Parameters
        ----------
        model : BaseEstimator
            The scikit-learn estimator to serialize.

        Returns
        -------
        Dict[str, Any]
            A dictionary representation of the model.
        """
        # Extract and build estimator params and its types/dtypes map
        params = model.get_params(deep=False)
        param_types, param_dtypes = self._get_type_maps(params)

        # Build serializable estimator including extra info
        serialized_estimator = {
            "estimator_class": model.__class__.__name__,
            "estimator_package": _package_of(type(model)),
            "params": self.convert_to_serializable(params),
            "param_types": param_types,
            "param_dtypes": param_dtypes,
        }

        try:
            check_is_fitted(model)
        except NotFittedError:
            return serialized_estimator

        # Extract and build fitted attributes and its types/dtypes map
        attributes = self._extract_estimator_attributes(model)
        attribute_types, attribute_dtypes = self._get_type_maps(attributes)

        serializable_attributes = self.convert_to_serializable(attributes)

        return {
            **serialized_estimator,
            "attributes": serializable_attributes,
            "attribute_types": attribute_types,
            "attribute_dtypes": attribute_dtypes,
        }

    @staticmethod
    def _resolve_package_version(name: str) -> str:
        """
        Best-effort version lookup for a top-level package name (e.g. "sklearn",
        "chemotools"). Prefers the already-imported module's own `__version__` attribute -
        the only option for a package like scikit-learn, whose import name ("sklearn")
        differs from its distribution name ("scikit-learn"), so it has no
        `importlib.metadata` entry under "sklearn" - falling back to package metadata for
        packages where the import name does match the distribution name. Returns "unknown"
        if neither resolves, mirroring `_openmodels_version`'s own fallback.
        """
        module = sys.modules.get(name)
        version = getattr(module, "__version__", None)
        if version:
            return str(version)
        try:
            return _package_version(name)
        except PackageNotFoundError:
            return "unknown"

    def _collect_package_names(self, serialized: Any, names: Set[str]) -> None:
        """
        Recursively walk an already-serialized estimator dict (params/attributes, however
        deeply nested) and collect the top-level package name of every estimator class found
        in it, from each node's own "estimator_package" - this is what lets a composite
        estimator mixing packages (e.g. a scikit-learn `Pipeline` with a third-party step)
        report every package involved, not just the outermost one. Nodes without the field
        fall back to the bare-name class registry.
        """
        if isinstance(serialized, dict):
            estimator_class = serialized.get("estimator_class")
            if estimator_class is not None:
                package = serialized.get("estimator_package")
                if package is None:
                    cls = self._by_name.get(estimator_class)
                    package = _package_of(cls) if cls is not None else None
                if package is not None:
                    names.add(package)
            for value in serialized.values():
                self._collect_package_names(value, names)
        elif isinstance(serialized, (list, tuple)):
            for item in serialized:
                self._collect_package_names(item, names)

    def serialize(self, model: BaseEstimator) -> Dict[str, Any]:
        """
        Serialize a scikit-learn estimator to a dictionary.

        This method extracts relevant attributes from the model, converts them to
        JSON-serializable types, and returns a dictionary representation of the model,
        with a "metadata" block (producer/format bookkeeping) attached at the root.

        Parameters
        ----------
        model : BaseEstimator
            The scikit-learn estimator to serialize.

        Returns
        -------
        Dict[str, Any]
            A dictionary representation of the model.

        Raises
        ------
        SerializationError
            If there's an error during serialization.

        Examples
        --------
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(n_samples=100, n_features=20, n_classes=2)
        >>> model = LogisticRegression().fit(X, y)
        >>> serializer = SklearnSerializer()
        >>> serialized_dict = serializer.serialize(model)
        """
        serialized_estimator = self._serialize_core(model)

        # The domain package is always a dependency, even when no class in the tree is its own.
        package_names: Set[str] = {"sklearn"}
        self._collect_package_names(serialized_estimator, package_names)
        packages = {
            name: self._resolve_package_version(name) for name in sorted(package_names)
        }

        metadata = {
            # ONNX-style: the tool that wrote this file, not the model's own package.
            "producer_name": "openmodels",
            "producer_version": _openmodels_version(),
            "domain": "sklearn",
            "domain_version": sklearn.__version__,
            "packages": packages,
            "openmodels_format_version": OPENMODELS_FORMAT_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dependency_versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
        }
        return {**serialized_estimator, "metadata": metadata}

    def deserialize(self, data: Dict[str, Any]) -> BaseEstimator:
        """
        Deserialize a dictionary representation back into a scikit-learn estimator.

        This method reconstructs a scikit-learn estimator from its dictionary
        representation, converting attributes back to their original types.

        Parameters
        ----------
        data : Dict[str, Any]
            The dictionary representation of the model.

        Returns
        -------
        BaseEstimator
            The deserialized scikit-learn estimator.

        Raises
        ------
        UnsupportedEstimatorError
            If the estimator class is not supported.

        Examples
        --------
        >>> serializer = SklearnSerializer()
        >>> deserialized_model = serializer.deserialize(serialized_dict)
        >>> predictions = deserialized_model.predict(X_test)
        """
        # Version control check. `data` itself is the fallback for pre-v2 files, which have
        # no nested "metadata" key and carried these fields flat at the top level instead.
        metadata = data.get("metadata", data)
        format_version = metadata.get("openmodels_format_version")
        if "domain_version" in metadata:
            sklearn_version = metadata["domain_version"]
        elif format_version is None or format_version <= 2:
            # v1/v2 files have no domain_version; their producer_version always held the
            # scikit-learn version. From v3 on it's the writing tool's version, so it must
            # never be compared against scikit-learn.
            sklearn_version = metadata.get("producer_version")
        else:
            sklearn_version = None
        self._check_version(sklearn_version)
        # v2 files called this map "producers".
        self._check_package_versions(
            metadata.get("packages", metadata.get("producers"))
        )
        self._check_format_version(format_version)

        self._ambiguity_warned = set()
        return self._deserialize_core(data)

    def _deserialize_core(self, data: Dict[str, Any]) -> BaseEstimator:
        """
        Reconstruct a scikit-learn estimator from its dictionary representation, without
        reading the root-only "metadata" block (see `deserialize`). This is the method used
        for every recursive/nested estimator dict.
        """
        # Reset per-call scratch state used by _deserialize_cfnode/_resolve_birch_leaf_links.
        self._birch_pending_leaf_links = []

        estimator_class = data["estimator_class"]
        if estimator_class in NOT_SUPPORTED_ESTIMATORS:
            raise UnsupportedEstimatorError(
                f"Unsupported estimator class: {estimator_class}"
            )

        # Reconstruct params with correct types/dtypes
        params = data.get("params", {})
        param_types = data.get("param_types", {})
        param_dtypes = data.get("param_dtypes", {})

        # Ensure tuples are reconstructed correctly
        for key, value in params.items():
            if param_types.get(key) == "tuple" and isinstance(value, list):
                params[key] = tuple(value)

        # Get valid constructor arguments for the estimator
        estimator_cls = self._resolve_class(data)
        valid_args = list(inspect.signature(estimator_cls.__init__).parameters.keys())
        # Remove 'self' if present
        valid_args = [arg for arg in valid_args if arg != "self"]

        reconstructed_params = {}
        for param_name, param_value in params.items():
            # Only include params that are valid constructor arguments
            if param_name not in valid_args:
                continue
            param_type = param_types.get(param_name)
            param_dtype = param_dtypes.get(param_name) or None
            reconstructed_params[param_name] = _restore_tuple_param(
                estimator_cls,
                param_name,
                self.convert_from_serializable(param_value, param_type, param_dtype),
            )
        model = estimator_cls(**reconstructed_params)

        if "attributes" not in data:
            return model  # Unfitted model

        # A neighbors estimator's search tree is rebuilt after the loop from the estimator's
        # own state; a stored one (which lacks the metric and leaf_size) isn't loaded then.
        rebuilds_tree = isinstance(model, NeighborsBase) and all(
            key in data["attributes"] for key in _NEIGHBORS_TREE_INPUTS
        )
        for attribute, value in data["attributes"].items():
            attr_type = data["attribute_types"].get(attribute)
            attr_dtype = data.get("attribute_dtypes", {}).get(attribute) or None

            if attribute == "_tree" and rebuilds_tree:
                continue
            # Handle tree_ separately
            if attr_type == "Tree":
                model.tree_ = self._deserialize_tree(value)
                continue
            # Search trees (a neighbors estimator's `_tree`, KernelDensity's `tree_`), restored
            # under the attribute's own name; their owners rebuild them after the loop.
            if attr_type in ("KDTree", "BallTree"):
                tree_cls = KDTree if attr_type == "KDTree" else BallTree
                setattr(
                    model, attribute, self._deserialize_search_tree(value, tree_cls)
                )
                continue
            # Use convert_from_serializable for all attributes
            setattr(
                model,
                attribute,
                self.convert_from_serializable(value, attr_type, attr_dtype),
            )

        if estimator_class == "Birch" and self._birch_pending_leaf_links:
            self._resolve_birch_leaf_links()
        self._rebuild_neighbors_tree(model)
        self._rebuild_kernel_density_tree(model)
        self._rebuild_derived_attributes(model)

        return model
