"""
Mixins for extensible serialization of Python, NumPy, and SciPy objects.

This module provides a flexible mixin-based architecture for converting complex objects
(such as slices, types, NumPy arrays, and SciPy structures) into JSON-serializable formats.
Each mixin implements handlers for specific types, allowing easy extension and modular
support for new serialization targets.
"""

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, csr_array, csc_array  # type: ignore
from scipy.interpolate import interp1d, BSpline  # type: ignore
from scipy.stats._distn_infrastructure import rv_continuous_frozen  # type: ignore
import scipy.stats  # type: ignore

from typing import Any, Optional, Callable, Dict


class SerializerMixin:
    """
    Base mixin providing recursive serialization and native Python object
    serialization with a dispatch mechanism. Other mixins only need to implement
    `_get_handlers()` and serialization helpers.
    """

    def convert_to_serializable(self, value):
        """Recursively convert values into JSON-serializable types."""
        # First check custom handlers
        for typ, handler in self._get_serializer_handlers():
            if isinstance(value, typ):
                return handler(value)

        # Recursive case: dict, list, tuple
        if isinstance(value, dict):
            if all(isinstance(k, str) for k in value):
                return {k: self.convert_to_serializable(v) for k, v in value.items()}
            # Non-string keys (e.g. int) can't be represented as JSON object keys without
            # losing their type, so fall back to a self-describing keys/values envelope that
            # _deserialize_dict can reconstruct exactly.
            return {
                "__openmodels_dict__": True,
                "keys": [self.convert_to_serializable(k) for k in value.keys()],
                "key_types": [type(k).__name__ for k in value.keys()],
                "values": [self.convert_to_serializable(v) for v in value.values()],
            }

        if isinstance(value, (list, tuple)):
            return [self.convert_to_serializable(v) for v in value]

        return value

    def convert_from_serializable(
        self, value: Any, value_type: Any = "none", value_dtype: Optional[str] = None
    ) -> Any:

        if isinstance(value_type, list) and isinstance(value, list):
            return [
                self.convert_from_serializable(v, t, value_dtype)
                for v, t in zip(value, value_type)
            ]

        if isinstance(value_type, str):
            for typ_name, handler in self._get_deserializer_handlers():
                if typ_name == value_type:
                    if value_type == "ndarray":
                        return handler(value, value_dtype)
                    return handler(value)

        return value

    # --- Python-native specific serializers/deserializers ---
    def _serialize_slice(self, value: slice):
        return {"start": value.start, "stop": value.stop, "step": value.step}

    def _deserialize_slice(self, value):
        return slice(
            start=value.get("start"),
            stop=value.get("stop"),
            step=value.get("step"),
        )

    def _serialize_type(self, value: type):
        return {"type_name": value.__name__}

    def _deserialize_type(self, value):
        # Deserialize Python type objects from their string name
        # Only allow a safe whitelist of types; default to float if not found
        allowed_types = {
            "int": int,
            "float": float,
            "str": str,
            "bool": bool,
            "tuple": tuple,
            "list": list,
            "dict": dict,
        }
        return allowed_types.get(value["type_name"], float)

    def _serialize_function(self, func: Callable) -> Dict[str, str]:
        """Serialize a Python function by its module and name."""
        return {
            "module": func.__module__,
            "name": func.__name__,
        }

    def _deserialize_function(self, data: Dict[str, str]) -> Callable:
        """Deserialize a Python function from its module and name."""
        module = __import__(data["module"], fromlist=[data["name"]])
        return getattr(module, data["name"])

    def _deserialize_dict(self, value: Any) -> Any:
        """Deserialize a dict, restoring non-string key types for the envelope produced by
        convert_to_serializable's dict branch. Plain string-keyed dicts (the common case,
        including dicts produced by older openmodels versions) pass through unchanged.
        """
        if isinstance(value, dict) and value.get("__openmodels_dict__"):
            allowed_key_types = {"int": int, "float": float, "bool": bool, "str": str}
            return {
                allowed_key_types.get(kt, lambda x: x)(
                    k
                ): self.convert_from_serializable(v)
                for k, kt, v in zip(value["keys"], value["key_types"], value["values"])
            }
        return value

    # --- Handlers ---
    def _get_serializer_handlers(self):
        """Each mixin extends this list."""
        return [
            (slice, self._serialize_slice),
            (type, self._serialize_type),
            (Callable, self._serialize_function),
        ]

    def _get_deserializer_handlers(self):
        return [
            ("bool", bool),
            ("float", float),
            ("int", int),
            ("slice", self._deserialize_slice),
            ("str", str),
            ("type", self._deserialize_type),
            ("tuple", tuple),
            ("dict", self._deserialize_dict),
            ("function", self._deserialize_function),
        ]


class NumpySerializerMixin(SerializerMixin):
    # --- Helpers ---
    def _get_dtype(self, value: Any) -> str:
        """
        Get the dtype of a numpy array, otherwise return empty string.
        """
        if isinstance(value, np.ndarray):
            return str(value.dtype)  # Get the actual numpy dtype
        elif isinstance(value, (list, tuple)) and value:
            # If it's a list/tuple that will become an ndarray, check its elements
            first_elem = value[0]
            if isinstance(first_elem, (int, np.integer)):
                return "int32"  # Use int32 for integer lists
            elif isinstance(first_elem, (float, np.floating)):
                return "float64"  # Use float64 for float lists
        return ""

    _POOLING_FUNC_REGISTRY = {
        "mean": np.mean,
        "median": np.median,
        "max": np.max,
        "min": np.min,
        "sum": np.sum,
    }

    # --- NumPy specific serializers/deserializers ---
    def _serialize_ndarray(self, value: np.ndarray):
        return self.convert_to_serializable(value.tolist())

    def _serialize_generic(self, value: np.generic):
        return value.item()

    def _deserialize_randomstate(self, value):
        rs = np.random.RandomState()
        rs.set_state(tuple(value))
        return rs

    def _serialize_numpy_function(self, value):
        # Only handle known numpy functions
        for name, func in self._POOLING_FUNC_REGISTRY.items():
            if value is func:
                return {"numpy_function": name}
        # fallback: use __name__ if possible
        return {"numpy_function": getattr(value, "__name__", None)}

    def _deserialize_numpy_function(self, value, value_dtype=None):
        name = value.get("numpy_function")
        if name in self._POOLING_FUNC_REGISTRY:
            return self._POOLING_FUNC_REGISTRY[name]
        raise ValueError(f"Unknown numpy function: {name}")

    # --- Handlers ---
    def _get_serializer_handlers(self):
        return [
            (np.ndarray, self._serialize_ndarray),
            (np.generic, self._serialize_generic),
            (np.dtype, str),
            (type(np.dtype("float64")), str),
            (
                np.random.RandomState,
                lambda v: [self.convert_to_serializable(x) for x in v.get_state()],
            ),
            (type(np.mean), self._serialize_numpy_function),
        ] + super()._get_serializer_handlers()

    def _get_deserializer_handlers(self):
        return [
            ("ndarray", lambda v, dt=None: np.array(v, dtype=(dt or None))),
            ("generic", lambda v: np.array(v).item()),
            ("float64", np.float64),
            ("int32", int),
            ("int64", int),
            ("dtype", np.dtype),
            ("Float64DType", np.dtype),
            ("RandomState", self._deserialize_randomstate),
            ("_ArrayFunctionDispatcher", self._deserialize_numpy_function),
        ] + super()._get_deserializer_handlers()


class ScipySerializerMixin(SerializerMixin):
    # --- SciPy specific serializers/deserializers  ---
    def _serialize_csr_matrix(self, value: csr_matrix):
        csr_value = csr_matrix(value)
        return {
            "data": self.convert_to_serializable(csr_value.data),
            "indptr": self.convert_to_serializable(csr_value.indptr.astype(np.int32)),
            "indices": self.convert_to_serializable(csr_value.indices.astype(np.int32)),
            "shape": self.convert_to_serializable(csr_value.shape),
            "dtype": str(csr_value.data.dtype),
        }

    def _deserialize_csr_matrix(self, value, value_dtype=None):
        dtype = value.get("dtype", None) or value_dtype or np.float64
        return csr_matrix(
            (
                np.array(value["data"], dtype=dtype),
                np.array(value["indices"], dtype=np.int32),
                np.array(value["indptr"], dtype=np.int32),
            ),
            shape=tuple(value["shape"]),
        )

    def _serialize_interp1d(self, value: interp1d):
        fill_value = getattr(value, "fill_value", np.nan)
        if isinstance(fill_value, np.ndarray):
            fill_value = self.convert_to_serializable(fill_value)
        return {
            "x": self.convert_to_serializable(value.x),
            "y": self.convert_to_serializable(value.y),
            "kind": getattr(value, "_kind", "linear"),
            "fill_value": fill_value,
            "bounds_error": getattr(value, "bounds_error", None),
            "assume_sorted": getattr(value, "assume_sorted", False),
            "axis": getattr(value, "axis", -1),
            "copy": getattr(value, "copy", True),
        }

    def _deserialize_interp1d(self, value, value_dtype=None):
        return interp1d(
            x=value["x"],
            y=value["y"],
            kind=value["kind"],
            fill_value=value["fill_value"],
            bounds_error=value["bounds_error"],
            assume_sorted=value["assume_sorted"],
            axis=value["axis"],
            copy=value["copy"],
        )

    def _serialize_scipy_dist(self, value: rv_continuous_frozen):
        return {"dist_name": value.dist.name, "args": value.args, "kwargs": value.kwds}

    def _deserialize_scipy_dist(self, value, value_dtype=None):
        dist = getattr(scipy.stats, value["dist_name"])
        return dist(*value["args"], **value["kwargs"])

    def _serialize_bspline(self, spline: BSpline) -> Dict[str, Any]:
        """
        Serialize a scipy.interpolate.BSpline object.
        """
        return {
            "t": spline.t.tolist(),  # Knots
            "c": spline.c.tolist(),  # Coefficients
            "k": spline.k,  # Degree
            "extrapolate": spline.extrapolate,
        }

    def _deserialize_bspline(self, data: Dict[str, Any]) -> BSpline:
        """
        Deserialize a dictionary back into a scipy.interpolate.BSpline object.
        """
        return BSpline(
            t=np.array(data["t"]),
            c=np.array(data["c"]),
            k=data["k"],
            extrapolate=data["extrapolate"],
        )

    # --- Handlers ---
    def _get_serializer_handlers(self):
        return [
            (BSpline, self._serialize_bspline),
            # csr_matrix is the only sparse container openmodels round-trips on the wire, but
            # any scipy sparse container (the older *_matrix family or the newer array-API
            # *_array family) is accepted here - _serialize_csr_matrix normalizes it to csr via
            # the csr_matrix(value) constructor, which accepts any sparse-like input.
            (
                (csr_matrix, csc_matrix, csr_array, csc_array),
                self._serialize_csr_matrix,
            ),
            (interp1d, self._serialize_interp1d),
            (rv_continuous_frozen, self._serialize_scipy_dist),
        ] + super()._get_serializer_handlers()

    def _get_deserializer_handlers(self):
        return [
            ("BSpline", self._deserialize_bspline),
            ("csr_matrix", self._deserialize_csr_matrix),
            ("interp1d", self._deserialize_interp1d),
            ("scipy_dist", self._deserialize_scipy_dist),
        ] + super()._get_deserializer_handlers()
