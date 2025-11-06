"""
Scikit-learn model serializer for the OpenModels library.

This module provides a serializer for scikit-learn models, allowing them to be
converted to and from dictionary representations.
"""

from typing import Any, Callable, Dict, List, Tuple, Type, Optional, Union
import numpy as np
import inspect

from ._custom_estimator import load_custom_estimators

import sklearn
from sklearn.calibration import _CalibratedClassifier, _SigmoidCalibration
from sklearn.cluster._birch import _CFNode
from sklearn.cluster._bisect_k_means import _BisectingTree
from sklearn.ensemble._hist_gradient_boosting.predictor import TreePredictor
from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
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
from sklearn.metrics._scorer import _CurveScorer
from sklearn.metrics import get_scorer_names, get_scorer
from sklearn.multiclass import _ConstantPredictor
from sklearn.tree._tree import Tree
from sklearn.base import BaseEstimator, check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.utils.discovery import all_estimators
from sklearn.neighbors import KDTree

from openmodels.exceptions import UnsupportedEstimatorError
from openmodels.protocols import ModelSerializer
from openmodels.serializers.base import (
    NumpySerializerMixin,
    ScipySerializerMixin,
)
import warnings

ConverterFunc = Callable[[Any], Any]

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

KERNEL_REGISTRY = [
    "RBF",
    "WhiteKernel",
    "Sum",
    "Product",
    "ConstantKernel",
    "DotProduct",
]

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

TESTED_VERSIONS = ["1.6.1", "1.7.2"]

NOT_SUPPORTED_ESTIMATORS: list[str] = [
    # Regressors: all regressors work!! Hurray!
    # Classifiers: all classifiers work!! Hurray!
    # Clusters: all clusters work!! Hurray!
    # Exceptions encountered during testing:
    # Transformers:
    "PatchExtractor",  # ValueError: not enough values to unpack (expected 3, got 2)
    # Others:
    "LocalOutlierFactor",  # AttributeError: This 'LocalOutlierFactor' has no attribute 'predict'
]


# Dictionary of attribute exceptions
ATTRIBUTE_EXCEPTIONS: Dict[str, List] = {
    # Regressors:
    "PLSRegression": ["_x_mean", "_predict_1d"],
    "SVR": [
        "_sparse",
        "_n_support",
        "_dual_coef_",
        "_intercept_",
        "_probA",
        "_probB",
        "_gamma",
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
    "CCA": ["_x_mean", "_predict_1d"],
    "GammaRegressor": ["_base_loss"],
    "PoissonRegressor": ["_base_loss"],
    "PLSCanonical": ["_x_mean", "_predict_1d"],
    "IsotonicRegression": ["f_"],
    "TransformedTargetRegressor": ["_training_dim"],
    # Clusters:
    "BisectingKMeans": ["_bisecting_tree", "_n_threads", "_X_mean"],
    "Birch": ["_subcluster_norms"],
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
    ],
    "KNeighborsClassifier": ["_fit_method", "_fit_X", "_y", "_tree"],
    "RadiusNeighborsClassifier": ["_fit_method", "_fit_X", "_y", "_tree"],
    "RidgeClassifier": ["_label_binarizer"],
    "RidgeClassifierCV": ["_label_binarizer"],
    "StackingClassifier": ["_label_encoder"],
    "SVC": [
        "_sparse",
        "_n_support",
        "_dual_coef_",
        "_intercept_",
        "_probA",
        "_probB",
        "_gamma",
    ],
    "TunedThresholdClassifierCV": ["_curve_scorer"],
    # Transformers:
    "ColumnTransformer": ["_columns", "_remainder"],
    "OneHotEncoder": [
        "_infrequent_enabled",
        "_drop_idx_after_grouping",
        "_n_features_outs",
    ],
    "OrdinalEncoder": ["_missing_indices", "_infrequent_enabled"],
    "KBinsDiscretizer": ["_encoder"],
    "KernelPCA": ["_centerer"],
    "KNNImputer": ["_mask_fit_X", "_valid_mask"],
    "KNeighborsTransformer": ["_fit_method", "_tree", "_fit_X"],
    "PowerTransformer": ["_scaler"],
    "RadiusNeighborsTransformer": ["_fit_method", "_tree"],
    "SimpleImputer": ["_fit_dtype"],
    "MiniBatchNMF": ["_n_components", "_transform_max_iter", "_beta_loss", "_gamma"],
    "MissingIndicator": ["_n_features", "_precomputed"],
    "MultiLabelBinarizer": ["_cached_dict"],
    "PolynomialFeatures": ["_max_degree", "_n_out_full", "_min_degree"],
    "PLSSVD": ["_x_mean", "_x_std"],
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
    ],
    "NearestNeighbors": ["_fit_method", "_tree", "_fit_X"],
    "TfidfVectorizer": ["_tfidf"],
}


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

        These estimators are merged into the serializer's internal registry for this instance only,
        allowing support for custom or external estimators without affecting the global registry.

    See Also
    --------
    scikit-learn developer guide:
        https://scikit-learn.org/stable/developers/develop.html

    sklearn.utils.discovery.all_estimators:
        https://scikit-learn.org/stable/modules/generated/sklearn.utils.discovery.all_estimators.html

    skltemplate.utils.discovery.all_estimators (project template):
        https://contrib.scikit-learn.org/project-template/generated/skltemplate.utils.discovery.all_estimators.html

    Developer Notes
    --------------
    For third-party packages compatible with scikit-learn, it is recommended to implement
    an ``all_estimators()`` utility following the scikit-learn API and template above.
    This enables automatic discovery and integration of custom estimators for serialization.

    If you are maintaining a scikit-learn compatible package, let us know!
    We are happy to extend our testing to include your estimators, ensuring everything works
    smoothly and that we cover any unique types or patterns used in your library.

    To request official support for your package, please open an issue at:
    https://github.com/SF-Tec/openmodels/issues

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
    ):
        extra = (
            load_custom_estimators(custom_estimators, ALL_ESTIMATORS)
            if custom_estimators
            else {}
        )
        self._all_estimators: Dict[str, Type] = {**ALL_ESTIMATORS, **extra}

    # --- Helpers ---
    def _check_version(self, stored_version: Optional[str]) -> None:
        """
        Check compatibility between stored scikit-learn version and the current environment.

        Parameters
        ----------
        stored_version : str
            The scikit-learn version recorded during serialization.

        Notes
        -----
        - Issues a warning if the stored version does not match the current version.
        - Mentions the baseline supported version (1.7.1).
        - Does nothing if no version is stored (for backward compatibility).
        """
        if not stored_version:
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

    def _get_nested_types(self, item: Any) -> Any:
        """
        Recursively determine the type of elements within nested lists.

        Parameters
        ----------
        item : Any
            The item to inspect for nested types.

        Returns
        -------
        Any
            A nested list representing the types of elements in the input item.

        Examples
        ---------

        [1, [1, 2, [1, 2, 3]], 2] -> ['int',['int','int','ndarray'],'int']

        """
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
            return tuple(self._get_nested_types(subitem) for subitem in item)

        # Handle lists
        if isinstance(item, list):
            return [self._get_nested_types(subitem) for subitem in item]

        elif isinstance(item, BaseEstimator):
            # For estimators, return their class name instead of just 'BaseEstimator'
            return item.__class__.__name__
        else:
            # Return the type name if it's not a list or it's an empty list
            return type(item).__name__

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

        attributes = {key: getattr(estimator, key) for key in attribute_keys}

        return attributes

    # --- Handlers ---
    def _get_serializer_handlers(self):
        # important to run before super() to deal with possible np.ndarray of estimators
        return [
            (BaseEstimator, self.serialize),
            (BaseLoss, self._serialize_loss),
            (KDTree, self._serialize_kdtree),
            (Kernel, self._serialize_kernel),
            (Tree, self._serialize_tree),
            (TreePredictor, self._serialize_tree_predictor),
            (np.ndarray, self._serialize_estimators_collection),
            (_CalibratedClassifier, self._serialize_calibrated_classifier),
            (_BisectingTree, self._serialize_bisecting_tree),
            (_CurveScorer, self._serialize_curve_scorer),
            (_CFNode, self._serialize_cfnode),
        ] + super()._get_serializer_handlers()

    def _get_deserializer_handlers(self):
        # Register losses
        loss_handlers = [
            (loss_name, (lambda v, ln=loss_name: self._deserialize_loss(v, ln)))
            for loss_name in LOSS_CLASS_REGISTRY.keys()
        ]
        # Estimators
        estimator_handlers = [
            (est_name, self.deserialize) for est_name in self._all_estimators.keys()
        ]

        kernel_handlers = [
            (kernel_name, self._deserialize_kernel) for kernel_name in KERNEL_REGISTRY
        ]
        return (
            [
                ("estimators_collection", self._deserialize_estimators_collection),
                ("TreePredictor", self._deserialize_tree_predictor),
                ("_BisectingTree", self._deserialize_bisecting_tree),
                ("_CalibratedClassifier", self._deserialize_calibrated_classifier),
                ("_CurveScorer", self._deserialize_curve_scorer),
                ("_CFNode", self._deserialize_cfnode),
            ]
            + kernel_handlers
            + loss_handlers
            + estimator_handlers
            + super()._get_deserializer_handlers()
        )

    # --- Sklearn specific serializers/deserializers ---
    def _serialize_bisecting_tree(self, tree: _BisectingTree) -> dict:
        return {
            "center": self.convert_to_serializable(tree.center),
            "indices": self.convert_to_serializable(tree.indices),
            "score": tree.score,
            "label": getattr(tree, "label", None),
            "left": self._serialize_bisecting_tree(tree.left) if tree.left else None,
            "right": self._serialize_bisecting_tree(tree.right) if tree.right else None,
        }

    def _deserialize_bisecting_tree(self, data: dict) -> _BisectingTree:
        if data is None:
            return None
        node = _BisectingTree(
            center=self.convert_from_serializable(data["center"]),
            indices=self.convert_from_serializable(data["indices"]),
            score=data["score"],
        )
        if data.get("label") is not None:
            node.label = data["label"]
        node.left = self._deserialize_bisecting_tree(data["left"])
        node.right = self._deserialize_bisecting_tree(data["right"])
        return node

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
        estimator = self.deserialize(data["estimator"])
        calibrators = [self.deserialize(c) for c in data["calibrators"]]
        classes = np.array(data["classes"])
        method = data["method"]
        return _CalibratedClassifier(
            estimator, calibrators, classes=classes, method=method
        )

    def _serialize_cfnode(self, node: _CFNode) -> Dict[str, Any]:
        """Recursively serialize a _CFNode."""
        return {
            "threshold": node.threshold,
            "branching_factor": node.branching_factor,
            "is_leaf": node.is_leaf,
            "n_features": node.n_features,
            # dtype=X.dtype,
        }

    def _deserialize_cfnode(self, data: dict) -> _CFNode:
        if data is None:
            return None
        node = _CFNode(
            threshold=data["threshold"],
            branching_factor=data["branching_factor"],
            is_leaf=data["is_leaf"],
            n_features=data["n_features"],
            dtype=np.float64,  # or use dtype from centroids if needed
        )
        return node

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

    def _serialize_kdtree(self, value: KDTree) -> Dict[str, Any]:
        """
        Serializes a KDTree object to a dictionary.
        """
        # For KDTree, we'll use a simpler approach - just serialize the essential data
        # and let the tree be reconstructed from the data
        data = np.array(value.data)
        return {
            "data": self.convert_to_serializable(data),
        }

    def _deserialize_kdtree(self, kdtree_data: Dict[str, Any]) -> KDTree:
        """
        Deserializes a dictionary representation of a KDTree back to a KDTree object.
        """
        data = np.array(kdtree_data["data"])

        # Create KDTree with data - the tree will be rebuilt automatically
        return KDTree(data)

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
                                self.deserialize(est)
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
                        self.deserialize(est)
                        if isinstance(est, dict) and "estimator_class" in est
                        else est
                    )
                    for est in value
                ]
        return value

    def _serialize_kernel(self, kernel: Kernel) -> Dict[str, Any]:
        """
        Recursively serialize a sklearn.gaussian_process.kernels.Kernel object.
        """
        kernel_type = type(kernel).__name__
        params = kernel.get_params(deep=False)
        # Recursively serialize kernel parameters that are also kernels
        serialized_params = {}
        for k, v in params.items():
            if isinstance(v, Kernel):
                serialized_params[k] = self._serialize_kernel(v)
            else:
                serialized_params[k] = v
        return {
            "kernel_type": kernel_type,
            "params": serialized_params,
        }

    def _deserialize_kernel(self, data: Dict[str, Any]) -> Kernel:
        """
        Recursively deserialize a kernel dict back to a Kernel object.
        """
        kernel_type = data["kernel_type"]
        params = data["params"]
        kernel_cls = getattr(
            __import__("sklearn.gaussian_process.kernels", fromlist=[kernel_type]),
            kernel_type,
        )
        deserialized_params = {}
        for k, v in params.items():
            if isinstance(v, dict) and "kernel_type" in v:
                deserialized_params[k] = self._deserialize_kernel(v)
            else:
                deserialized_params[k] = v
        return kernel_cls(**deserialized_params)

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

    def serialize(self, model: BaseEstimator) -> Dict[str, Any]:
        """
        Serialize a scikit-learn estimator to a dictionary.

        This method extracts relevant attributes from the model, converts them to
        JSON-serializable types, and returns a dictionary representation of the model.

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
        # Extract and build estimator params and its types/dtypes map
        params = model.get_params(deep=False)
        param_types, param_dtypes = self._get_type_maps(params)

        # Build serializable estimator including extra info
        serialized_estimator = {
            "estimator_class": model.__class__.__name__,
            "params": self.convert_to_serializable(params),
            "param_types": param_types,
            "param_dtypes": param_dtypes,
            "producer_version": sklearn.__version__,
            "producer_name": model.__module__.split(".")[0],
            "domain": "sklearn",
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
        # Version control check
        self._check_version(data.get("producer_version"))

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
        estimator_cls = self._all_estimators[estimator_class]
        valid_args = list(inspect.signature(estimator_cls.__init__).parameters.keys())
        # Remove 'self' if present
        valid_args = [arg for arg in valid_args if arg != "self"]

        reconstructed_params = {}
        for param_name, param_value in params.items():
            # Only include params that are valid constructor arguments
            if param_name not in valid_args:
                continue
            # Handle PatchExtractor's 'patch_size' parameter
            if (
                estimator_class == "PatchExtractor"
                and param_name == "patch_size"
                and isinstance(param_value, list)
            ):
                param_value = tuple(param_value)
            param_type = param_types.get(param_name)
            param_dtype = param_dtypes.get(param_name) or None
            reconstructed_params[param_name] = self.convert_from_serializable(
                param_value, param_type, param_dtype
            )
        model = estimator_cls(**reconstructed_params)

        if "attributes" not in data:
            return model  # Unfitted model

        for attribute, value in data["attributes"].items():
            attr_type = data["attribute_types"].get(attribute)
            attr_dtype = data.get("attribute_dtypes", {}).get(attribute) or None

            # Handle tree_ separately
            if attr_type == "Tree":
                model.tree_ = self._deserialize_tree(value)
                continue
            # Skip _tree attribute for KDTree - let the transformer recreate it
            if attr_type == "KDTree":
                model._tree = self._deserialize_kdtree(value)
                continue
            # Use convert_from_serializable for all attributes
            setattr(
                model,
                attribute,
                self.convert_from_serializable(value, attr_type, attr_dtype),
            )

        return model
