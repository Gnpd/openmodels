"""
Mixins for extensible serialization of Python, NumPy, and SciPy objects.

This module provides a flexible mixin-based architecture for converting complex objects
(such as slices, types, NumPy arrays, and SciPy structures) into JSON-serializable formats.
Each mixin implements handlers for specific types, allowing easy extension and modular
support for new serialization targets.
"""

import ast
import importlib
import inspect
import sys
import warnings
from collections import UserList
from types import ModuleType

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, csr_array, csc_array  # type: ignore
from scipy.interpolate import interp1d, BSpline  # type: ignore
from scipy.stats._distn_infrastructure import rv_frozen  # type: ignore
import scipy.stats  # type: ignore

from typing import Any, Optional, Callable, Dict, Set

from openmodels.exceptions import DeserializationError, SerializationError

# Top-level packages whose (already-imported) functions a model file may reference, e.g. a
# SelectKBest score_func such as sklearn.feature_selection.chi2.
DEFAULT_FUNCTION_ROOTS = frozenset({"numpy", "scipy", "sklearn"})

# Type of NumPy's array functions (np.mean, np.std, np.linalg.norm, ...).
_NUMPY_ARRAY_FUNCTION_TYPE = type(np.mean)

# Modules searched for a function that doesn't record its own `__module__` (SciPy's ufuncs,
# e.g. scipy.special.expit). Only already-imported modules are searched.
_FUNCTION_MODULE_CANDIDATES = ("numpy", "scipy.special")


def _find_function_module(func: Any) -> Optional[str]:
    """Name of the module (among `_FUNCTION_MODULE_CANDIDATES`) that exposes `func` under its
    own `__name__`, or None."""
    name = getattr(func, "__name__", None)
    if not isinstance(name, str):
        return None
    for module_name in _FUNCTION_MODULE_CANDIDATES:
        if getattr(sys.modules.get(module_name), name, None) is func:
            return module_name
    return None


# Keys of a saved slice (see SerializerMixin._serialize_slice).
_SLICE_KEYS = frozenset({"start", "stop", "step"})

# Keys of a saved frozen SciPy distribution (see ScipySerializerMixin._serialize_scipy_dist).
_SCIPY_DIST_KEYS = frozenset({"dist_name", "args", "kwargs"})

# Python builtin types a type-valued param (e.g. `dtype=float`) may name; NumPy scalar types
# are resolved separately (see SerializerMixin._deserialize_type).
_BUILTIN_TYPES: Dict[str, type] = {
    t.__name__: t
    for t in (int, float, bool, str, bytes, complex, tuple, list, dict, object)
}

# NumPy's integer and float scalar types by name (int8 ... uint64, float16 ... float64, ...),
# for scalar values, which are tagged with `type(value).__name__` and written as their
# `.item()`. Some names depend on the platform (np.longlong is "int64" on Windows and
# "longlong" on Linux), so the C names are added for files written elsewhere. np.bool_ values
# are tagged "bool_" (see SklearnSerializer._get_nested_types), as NumPy 2 names it "bool".
_NUMPY_SCALAR_TYPES: Dict[str, type] = {
    t.__name__: t
    for t in (
        np.dtype(code).type
        for code in np.typecodes["AllInteger"] + np.typecodes["Float"]
    )
}
for _name in ("intc", "uintc", "longlong", "ulonglong", "longdouble"):
    _NUMPY_SCALAR_TYPES.setdefault(_name, getattr(np, _name))
_NUMPY_SCALAR_TYPES["bool_"] = np.bool_


def _key_to_text(key: Any) -> str:
    """
    A dict key as the text JSON requires for object keys: strings unchanged; ints, floats,
    bools, None and NumPy scalars as `str()`. Their type is recorded in the dict's type entry
    ("key_types") and restored by `SerializerMixin._text_to_key`. Dicts in values written
    without a type entry (kernels, Birch nodes, tree predictors, ...) only ever have string
    keys. Other keys (tuples, ...) can't be written.
    """
    if isinstance(key, str):
        return key
    if isinstance(key, (np.integer, np.floating, np.bool_)):
        key = key.item()
    if key is None or isinstance(key, (int, float)):  # bool is an int
        return str(key)
    raise SerializationError(
        f"Can't serialize dict key {key!r} of type {type(key).__name__}: only str, int, "
        f"float, bool and None keys are supported"
    )


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
            # Keys are written as text, as JSON requires; a non-string key's type is recorded
            # in the dict's type entry ("key_types", see SklearnSerializer._get_dict_types).
            serialized = {
                _key_to_text(k): self.convert_to_serializable(v)
                for k, v in value.items()
            }
            if len(serialized) != len(value):
                raise SerializationError(
                    f"Can't serialize a dict whose keys have the same text "
                    f"(e.g. 1 and '1'): {list(value)!r}"
                )
            return serialized

        if isinstance(value, (list, tuple)):
            return [self.convert_to_serializable(v) for v in value]

        # A UserList isn't a list subclass; it's saved, and loads, as its plain list. E.g.
        # scikit-learn 1.6's ColumnTransformer keeps its remainder columns in one that warns
        # when indexed, so its `.data` is read directly.
        if isinstance(value, UserList):
            return self.convert_to_serializable(value.data)

        return value

    def convert_from_serializable(
        self, value: Any, value_type: Any = "none", value_dtype: Optional[str] = None
    ) -> Any:

        # Type maps hold tuples for tuple values (e.g. a Pipeline step's ("str", "<Class>"))
        # when the dict stays in memory or goes through pickle; JSON-like formats have already
        # turned them into lists. Both describe the same nested structure.
        if isinstance(value_type, (list, tuple)) and isinstance(value, (list, tuple)):
            # A list inside a typed dict has one dtype per element (see the "dict" case below).
            value_dtypes = (
                value_dtype
                if isinstance(value_dtype, list)
                else [value_dtype] * len(value)
            )
            return [
                self.convert_from_serializable(v, t, d)
                for v, t, d in zip(value, value_type, value_dtypes)
            ]

        # Typed containers, written for dicts whose values or keys need restoring: {"dict":
        # {key: type}} (plus "key_types" for non-string keys) with a dtype dict mirroring it,
        # and {"tuple": [types]} for a tuple inside one.
        if isinstance(value_type, dict):
            if (
                isinstance(value_type.get("dict"), dict)
                and set(value_type) <= {"dict", "key_types"}
                and isinstance(value, dict)
            ):
                return self._deserialize_typed_dict(
                    value,
                    value_type["dict"],
                    value_dtype,
                    value_type.get("key_types"),
                )
            if (
                len(value_type) == 1
                and isinstance(value_type.get("tuple"), list)
                and isinstance(value, list)
            ):
                return tuple(
                    self.convert_from_serializable(
                        value, value_type["tuple"], value_dtype
                    )
                )

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
        # slice() takes positional arguments only.
        return slice(value.get("start"), value.get("stop"), value.get("step"))

    def _serialize_type(self, value: type):
        return {"type_name": value.__name__}

    def _deserialize_type(self, value):
        """
        Resolve a type written by `_serialize_type` (its `__name__`, e.g. a `dtype=np.int64`
        param) back to the type: a Python builtin from `_BUILTIN_TYPES`, else a NumPy scalar
        type. Only canonical NumPy names (`np.dtype(name).type.__name__ == name`) are
        accepted, since that's all the writer produces; aliases like "f8" and structured specs
        like "i4,f8" are not. `np.dtype` only parses the name - nothing is imported or called.
        NumPy's abstract scalar types (`np.number`, `np.integer`, ...), which aren't dtypes, are
        resolved as `np.<name>` when that is an `np.generic` subclass with that exact name.
        An unknown name falls back to `float`, as before, but with a warning.
        """
        name = value.get("type_name")
        if name in _BUILTIN_TYPES:
            return _BUILTIN_TYPES[name]
        try:
            numpy_type = np.dtype(name).type
        except (TypeError, ValueError):
            numpy_type = None
        if (
            numpy_type is not None
            and numpy_type is not np.void
            and numpy_type.__name__ == name
        ):
            return numpy_type
        abstract_type = (
            getattr(np, name, None)
            if isinstance(name, str) and not name.startswith("_")
            else None
        )
        if (
            isinstance(abstract_type, type)
            and issubclass(abstract_type, np.generic)
            and abstract_type is not np.void
            and abstract_type.__name__ == name
        ):
            return abstract_type
        warnings.warn(
            f"Unknown type '{name}' in the model file; loading it as float. The loaded "
            f"model may behave differently from the original.",
            UserWarning,
        )
        return float

    def _serialize_function(self, func: Callable) -> Dict[str, str]:
        """Serialize a Python function by its module and name."""
        name = getattr(func, "__name__", None)
        module = getattr(func, "__module__", None) or _find_function_module(func)
        if not isinstance(name, str) or module is None:
            raise SerializationError(
                f"Can't serialize callable {func!r}: it has no importable module and name"
            )
        # A bound method can only be saved by name when its module exposes that very method
        # (np.random.rand is the global RandomState's); otherwise the object it's bound to,
        # e.g. a fitted scaler's fit_transform, would be lost.
        if (
            inspect.ismethod(func)
            and getattr(sys.modules.get(module), name, None) is not func
        ):
            raise SerializationError(
                f"Can't serialize bound method {func!r}: only functions a module exposes by "
                f"name can be saved, not methods of an object"
            )
        return {"module": module, "name": name}

    def _allowed_function_roots(self) -> Set[str]:
        """Top-level packages whose already-imported modules may provide deserialized
        functions. Subclasses extend this (e.g. with the packages of registered estimators).
        """
        return set(DEFAULT_FUNCTION_ROOTS)

    def _is_trusted_function_module(self, module_name: str) -> bool:
        """Whether the user explicitly trusted this module, allowing it to be imported.
        Nothing is trusted by default."""
        return False

    def _deserialize_function(self, data: Dict[str, str]) -> Callable:
        """
        Deserialize a Python function from its module and name.

        The file names the module, so it is untrusted: the function is only looked up in a
        module that is already imported and belongs to an allowed top-level package
        (`_allowed_function_roots`), and only modules the user explicitly trusted
        (`_is_trusted_function_module`) may be imported. Importing an arbitrary module would
        run its import-time side effects. Private names, ``__main__`` modules and anything
        that isn't a plain function, builtin, NumPy ufunc or a bound method the module exposes
        by name (np.random.rand) are refused.
        """
        module_name, name = data.get("module"), data.get("name")
        obj = None
        if (
            isinstance(module_name, str)
            and isinstance(name, str)
            and not name.startswith("_")
            and "__main__" not in module_name.split(".")
        ):
            module: Optional[ModuleType] = None
            if self._is_trusted_function_module(module_name):
                module = importlib.import_module(module_name)
            elif module_name.split(".")[0] in self._allowed_function_roots():
                module = sys.modules.get(module_name)
            obj = getattr(module, name, None) if module is not None else None
        if not (
            inspect.isfunction(obj)
            or inspect.isbuiltin(obj)
            or isinstance(obj, np.ufunc)
            or inspect.ismethod(obj)
        ):
            raise DeserializationError(
                f"function '{module_name}.{name}' is not allowed; pass "
                f"trusted_function_modules=[...] to permit it"
            )
        return obj

    def _deserialize_typed_dict(
        self,
        value: Dict[str, Any],
        value_types: Dict[str, Any],
        value_dtypes: Any,
        key_types: Optional[Dict[str, str]] = None,
    ) -> Dict[Any, Any]:
        """Restore each value of a dict from its own type (and dtype, for arrays), and each
        key listed in `key_types` from its text. A key without a type falls back to
        `_restore_dict_value`."""
        dtypes = value_dtypes if isinstance(value_dtypes, dict) else {}
        key_types = key_types if isinstance(key_types, dict) else {}
        return {
            (self._text_to_key(k, key_types[k]) if k in key_types else k): (
                self.convert_from_serializable(v, value_types[k], dtypes.get(k))
                if k in value_types
                else self._restore_dict_value(v)
            )
            for k, v in value.items()
        }

    def _text_to_key(self, text: str, key_type: str) -> Any:
        """Inverse of `_key_to_text` for a key of type `key_type` (e.g. "int", "float64",
        "bool", "NoneType"): the text is parsed, then restored with that type's handler.
        """
        try:
            if key_type == "NoneType" and text == "None":
                return None
            if key_type in ("bool", "bool_") and text in ("True", "False"):
                parsed: Any = text == "True"
            elif key_type.startswith(("int", "uint")):
                parsed = int(text)
            elif key_type.startswith("float"):
                parsed = float(text)
            else:
                raise ValueError("unsupported key type")
        except (AttributeError, TypeError, ValueError) as e:
            raise DeserializationError(
                f"Invalid dict key {text!r} of type {key_type!r} in the model file"
            ) from e
        return self.convert_from_serializable(parsed, key_type)

    def _restore_dict_value(self, value: Any) -> Any:
        """Hook for restoring a value inside a plain string-keyed dict, whose values carry no
        type tags of their own, recognised by its exact saved shape: slices here (e.g.
        ColumnTransformer's output_indices_), more in mixins (see ScipySerializerMixin).
        Anything else is returned unchanged. Since format v4, dicts whose values need
        restoring are typed per key instead; this is for files written before that."""
        if isinstance(value, dict) and set(value) == _SLICE_KEYS:
            return self._deserialize_slice(value)
        return value

    def _deserialize_dict(self, value: Any) -> Any:
        """Deserialize a dict tagged "dict". Files written before format v4 saved dicts with
        non-string keys as an `__openmodels_dict__` keys/values envelope, rebuilt here with
        its key types (newer files type such keys in the dict's type entry instead, see
        `_deserialize_typed_dict`). Values of plain string-keyed dicts go through
        `_restore_dict_value`, which leaves them unchanged unless a mixin recognises them.
        """
        if isinstance(value, dict) and value.get("__openmodels_dict__"):
            allowed_key_types = {"int": int, "float": float, "bool": bool, "str": str}
            return {
                allowed_key_types.get(kt, lambda x: x)(
                    k
                ): self.convert_from_serializable(v)
                for k, kt, v in zip(value["keys"], value["key_types"], value["values"])
            }
        if isinstance(value, dict):
            return {k: self._restore_dict_value(v) for k, v in value.items()}
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
            # NumPy ufuncs (np.log1p, scipy.special.expit) and C builtins (abs) are tagged by
            # their own type name, and bound methods (np.random.rand) "method";
            # _deserialize_function accepts them under the allowlist.
            ("ufunc", self._deserialize_function),
            ("builtin_function_or_method", self._deserialize_function),
            ("method", self._deserialize_function),
        ]


class NumpySerializerMixin(SerializerMixin):
    # --- Helpers ---
    def _get_dtype(self, value: Any) -> Any:
        """
        Get the dtype of a numpy array, otherwise return empty string. A list of arrays gets
        one dtype per element (see `_get_element_dtypes`).
        """
        if isinstance(value, np.ndarray):
            return str(value.dtype)  # Get the actual numpy dtype
        elif isinstance(value, (list, tuple)) and value:
            # If it's a list/tuple that will become an ndarray, check its elements
            first_elem = value[0]
            if isinstance(first_elem, np.ndarray):
                return self._get_element_dtypes(value)
            if isinstance(first_elem, (int, np.integer)):
                return "int32"  # Use int32 for integer lists
            elif isinstance(first_elem, (float, np.floating)):
                return "float64"  # Use float64 for float lists
        return ""

    def _get_element_dtypes(self, value: Any) -> Any:
        """
        One dtype per element of a list of arrays (e.g. MLP's `coefs_`), None for non-array
        elements, so a float32 model isn't widened to float64 on load. Written only when an
        array's values alone would rebuild it with another dtype (float32, int8, uint*, ...);
        otherwise "", as before, because openmodels 0.2.2 can't read a dtype list. Arrays of
        other kinds (object, strings) are rebuilt from their values, as before.
        """

        def inferred(array: np.ndarray) -> np.dtype:
            # What np.array() makes of the array's values: float64 for floats, the default
            # int for ints, bool for bools.
            return (
                np.array(array.ravel()[:1].tolist()).dtype
                if array.size
                else np.dtype(float)
            )

        if not any(
            isinstance(v, np.ndarray)
            and v.dtype.kind in "biuf"
            and v.dtype != inferred(v)
            for v in value
        ):
            return ""
        return [str(v.dtype) if isinstance(v, np.ndarray) else None for v in value]

    # --- NumPy specific serializers/deserializers ---
    def _serialize_ndarray(self, value: np.ndarray):
        return self.convert_to_serializable(value.tolist())

    def _deserialize_ndarray(self, value, value_dtype=None):
        """
        Rebuild an array from its nested lists and the `str(dtype)` written next to it. A
        structured dtype (e.g. HDBSCAN's `_single_linkage_tree_`) is written as a list of
        `(name, type)` tuples, which is a plain literal: it's parsed with `ast.literal_eval`
        (never `eval`), and each row goes back to a tuple, as NumPy requires for records.
        Only 1-D structured arrays are supported, which is what scikit-learn stores.
        """
        if value_dtype and value_dtype.startswith("["):
            try:
                dtype = np.dtype(ast.literal_eval(value_dtype))
            except (ValueError, TypeError, SyntaxError) as e:
                raise DeserializationError(
                    f"Invalid structured dtype {value_dtype!r}: {e}"
                ) from e
            return np.array([tuple(row) for row in value], dtype=dtype)
        return np.array(value, dtype=(value_dtype or None))

    def _serialize_generic(self, value: np.generic):
        return value.item()

    def _deserialize_randomstate(self, value):
        rs = np.random.RandomState()
        rs.set_state(tuple(value))
        return rs

    def _serialize_numpy_function(self, value):
        # NumPy array functions (np.std, np.linalg.norm, ...). The module is recorded because
        # the name alone can't find submodule functions (and np.fft is a module, not np.fft.fft).
        name = getattr(value, "__name__", None)
        if not isinstance(name, str):
            raise SerializationError(
                f"Can't serialize NumPy function {value!r}: no name"
            )
        return {"numpy_function": name, "module": value.__module__}

    def _deserialize_numpy_function(self, value, value_dtype=None):
        """
        Resolve a NumPy array function by name, from its recorded module (`"numpy"` for files
        written before the module was recorded). Only already-imported `numpy` modules are
        searched (nothing is imported), private names are refused, and the result must be a
        NumPy array function, which rules out modules (e.g. `np.fft`), classes and the like.
        """
        name = value.get("numpy_function")
        module_name = value.get("module", "numpy")
        obj = None
        if (
            isinstance(name, str)
            and not name.startswith("_")
            and isinstance(module_name, str)
            and module_name.split(".")[0] == "numpy"
        ):
            obj = getattr(sys.modules.get(module_name), name, None)
        if not isinstance(obj, _NUMPY_ARRAY_FUNCTION_TYPE):
            raise DeserializationError(
                f"Unknown NumPy function '{module_name}.{name}' in the model file"
            )
        return obj

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
            ("ndarray", self._deserialize_ndarray),
            ("generic", lambda v: np.array(v).item()),
            *_NUMPY_SCALAR_TYPES.items(),
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

    def _serialize_scipy_dist(self, value: rv_frozen):
        # Continuous and discrete frozen distributions (uniform(0, 1), randint(1, 10), ...).
        return {
            "dist_name": value.dist.name,
            "args": self.convert_to_serializable(value.args),
            "kwargs": self.convert_to_serializable(value.kwds),
        }

    def _deserialize_scipy_dist(self, value, value_dtype=None):
        """
        Rebuild a frozen distribution by calling the named `scipy.stats` distribution generator
        with the stored args. The name comes from the file, so it must name a distribution
        generator (`rv_continuous`/`rv_discrete` instance), never any other `scipy.stats`
        callable (e.g. `describe`), and private names are refused.
        """
        name = value.get("dist_name")
        generator = (
            getattr(scipy.stats, name, None)
            if isinstance(name, str) and not name.startswith("_")
            else None
        )
        if not isinstance(
            generator, (scipy.stats.rv_continuous, scipy.stats.rv_discrete)
        ):
            raise DeserializationError(
                f"Unknown distribution '{name}' in the model file"
            )
        return generator(*value.get("args", []), **value.get("kwargs", {}))

    def _restore_dict_value(self, value: Any) -> Any:
        # Distributions sit inside dict params (RandomizedSearchCV's param_distributions),
        # whose values have no type tags; they're recognised by their exact saved shape.
        if isinstance(value, dict) and set(value) == _SCIPY_DIST_KEYS:
            return self._deserialize_scipy_dist(value)
        return super()._restore_dict_value(value)

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
            (rv_frozen, self._serialize_scipy_dist),
        ] + super()._get_serializer_handlers()

    def _get_deserializer_handlers(self):
        return [
            ("BSpline", self._deserialize_bspline),
            ("csr_matrix", self._deserialize_csr_matrix),
            ("interp1d", self._deserialize_interp1d),
            ("rv_continuous_frozen", self._deserialize_scipy_dist),
            ("rv_discrete_frozen", self._deserialize_scipy_dist),
        ] + super()._get_deserializer_handlers()
