"""
This file contains functions for testing the serialization and deserialization of scikit-learn
models using the `openmodels` library.
"""

import functools
import os
from contextlib import contextmanager
from typing import (
    Iterator,
    Optional,
    Type,
    Union,
    Protocol,
    runtime_checkable,
    TypeVar,
    cast,
)
import numpy as np
from numpy import testing
from sklearn.base import BaseEstimator, clone
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_selection import GenericUnivariateSelect, f_classif
from sklearn.preprocessing import LabelBinarizer
from scipy.sparse import csr_matrix  # type: ignore

from openmodels import SerializationManager, SklearnSerializer


def assert_sparse_matrix_equal(a, b):
    """Assert that two sparse matrices are equal in shape and values (as float64)."""
    assert type(a) is type(b)
    assert a.shape == b.shape
    np.testing.assert_allclose(
        a.toarray().astype(np.float64), b.toarray().astype(np.float64)
    )


@runtime_checkable
class PredictorModel(Protocol):
    def predict(self, X: Union[np.ndarray, csr_matrix]) -> np.ndarray: ...


@runtime_checkable
class TransformerModel(Protocol):
    def transform(self, X: Union[np.ndarray, csr_matrix]) -> np.ndarray: ...


@runtime_checkable
class FittableModel(Protocol):
    def fit(
        self, X: Union[np.ndarray, csr_matrix], y: np.ndarray
    ) -> "FittableModel": ...


ModelType = Union[PredictorModel, TransformerModel, FittableModel]
T = TypeVar("T", bound=Union[BaseEstimator, ModelType])

_ROUNDTRIPPABLE_METHODS = ("fit", "fit_transform", "fit_predict")


@contextmanager
def roundtrip_fit(
    *classes: Type[BaseEstimator], format_name: str = "json"
) -> Iterator[None]:
    """
    Monkeypatch fit()/fit_transform()/fit_predict() on the given estimator classes so that,
    immediately after fitting, the estimator's fitted state is replaced with that of an
    openmodels serialize -> deserialize round-trip. Restores the original methods on exit.

    Only methods defined directly on the class are patched (not inherited ones), since e.g.
    TransformerMixin.fit_transform already calls self.fit(...) and would be covered by patching
    fit alone; only classes that override fit_transform/fit_predict directly need those patched.

    The fitted state is every instance attribute the fit call created or replaced: each is set
    to the loaded copy's value, or **deleted** if the loaded copy doesn't have it. So an
    attribute openmodels doesn't save is missing afterwards, exactly as on a really loaded
    model, and any later method that needs it fails. Two kinds of attribute are kept as they
    are:

    - constructor parameters: JSON has no tuple type, so e.g. a tuple-valued param would come
      back as a list, and copying it over would make the *test infrastructure* look like a
      fidelity bug in checks like check_dont_overwrite_parameters;
    - attributes the fit call left untouched (the same object before and after), such as the
      `set_output` configuration, metadata requests, or a parent estimator's callback context.
      An attribute fit mutates in place without rebinding counts as untouched, so this can't
      see it.

    Nested estimators held in parameters (e.g. a Pipeline's steps) are fitted in place and are
    only round-tripped if their own class is passed too.

    This lets sklearn's own estimator-specific test suites be reused unmodified against
    round-tripped models: any assertion they make about a fitted estimator becomes, implicitly,
    an openmodels fidelity check.
    """
    serializer = SklearnSerializer()
    manager = SerializationManager(serializer)
    originals: dict = {}

    def _roundtrip(instance, before_fit: dict) -> None:
        serialized = manager.serialize(instance, format_name=format_name)
        loaded = vars(manager.deserialize(serialized, format_name=format_name))
        params = instance.get_params(deep=False)
        for key, value in list(vars(instance).items()):
            if key in params or (key in before_fit and before_fit[key] is value):
                continue
            if key in loaded:
                setattr(instance, key, loaded[key])
            else:
                delattr(instance, key)
        for key, value in loaded.items():
            if key not in params and key not in vars(instance):
                setattr(instance, key, value)

    class _RoundtripDescriptor:
        """
        Wraps `orig` (a plain function, or a descriptor like sklearn's @available_if or a
        property) so attribute access still goes through __get__ first. This matters because
        many sklearn meta-estimators (Pipeline, FeatureAgglomeration, ...) use
        hasattr(estimator, "fit_transform")-style availability checks that rely on the
        descriptor's __get__ raising AttributeError when the method isn't actually available
        (e.g. a Pipeline whose last step has no transform()). Replacing the class attribute
        with a plain function would make it unconditionally "available", breaking that check
        for everyone who does hasattr() introspection - not just this test suite.
        """

        def __init__(self, orig):
            self._orig = orig

        def __get__(self, obj, objtype=None):
            if obj is None:
                return self
            # Raises AttributeError here (not inside the call below) if unavailable, exactly
            # like normal attribute access on the original descriptor would.
            bound_orig = self._orig.__get__(obj, objtype)

            @functools.wraps(bound_orig)
            def wrapper(*args, **kwargs):
                before_fit = dict(vars(obj))
                result = bound_orig(*args, **kwargs)
                _roundtrip(obj, before_fit)
                return result

            return wrapper

    for cls in classes:
        for method_name in _ROUNDTRIPPABLE_METHODS:
            if method_name in vars(cls):
                orig = vars(cls)[method_name]
                originals[(cls, method_name)] = orig
                setattr(cls, method_name, _RoundtripDescriptor(orig))

    try:
        yield
    finally:
        for (cls, method_name), orig in originals.items():
            setattr(cls, method_name, orig)


def ensure_correct_sparse_format(
    x: Union[np.ndarray, csr_matrix],
) -> Union[np.ndarray, csr_matrix]:
    """
    Ensure the input data is in the correct format for SVM models.
    """
    if isinstance(x, csr_matrix):
        # Create a new sparse matrix with correct dtypes
        data = x.data.astype(np.float64)
        indices = x.indices.astype(np.int32)
        indptr = x.indptr.astype(np.int32)

        # Ensure the indices are properly sorted
        for i in range(x.shape[0]):
            start = indptr[i]
            end = indptr[i + 1]
            if end - start > 0:  # If row is not empty
                order = np.argsort(indices[start:end])
                indices[start:end] = indices[start:end][order]
                data[start:end] = data[start:end][order]

        return csr_matrix(
            (data, indices, indptr),
            shape=x.shape,
            copy=False,  # Avoid unnecessary data copying
        )
    return x


def fit_model(
    model: FittableModel,
    x: Union[np.ndarray, csr_matrix],
    y: np.ndarray,
    abs: bool = False,
) -> FittableModel:
    """
    Fits a model to the provided data.

    Parameters
    ----------
    model : FittableModel
        The scikit-learn model to fit.
    x : Union[np.ndarray, csr_matrix]
        The training input samples.
    y : np.ndarray
        The target values (class labels in classification, real numbers in regression).
    abs : bool, default=False
        Whether to take the absolute value of the input data before fitting the model.

    Returns
    -------
    FittableModel
        The fitted scikit-learn model.
    """
    if not isinstance(model, FittableModel):
        raise TypeError("Model must have a 'fit' method")

    # Special case for LabelBinarizer
    if isinstance(model, LabelBinarizer):
        model.fit(y)
        return model

    if abs:
        if isinstance(x, csr_matrix):
            # Handle absolute value for sparse matrix
            x_abs = csr_matrix(
                (np.absolute(x.data), x.indices, x.indptr), shape=x.shape
            )
            model.fit(x_abs, y)
        else:
            model.fit(np.absolute(x), y)
    else:
        model.fit(x, y)
    return model


def run_test_predictions(
    model1: Union[PredictorModel, TransformerModel],
    model2: Union[PredictorModel, TransformerModel],
    x: Union[np.ndarray, csr_matrix],
    abs: bool = False,
) -> None:
    """
    Test if two models produce the same predictions.
    """
    # Always ensure input data is in correct format, even for dense arrays
    x = ensure_correct_sparse_format(x)

    if isinstance(model1, PredictorModel) and isinstance(model2, PredictorModel):
        if abs:
            if isinstance(x, csr_matrix):
                x_abs = csr_matrix(
                    (
                        np.absolute(x.data),
                        x.indices.astype(np.int32),
                        x.indptr.astype(np.int32),
                    ),
                    shape=x.shape,
                )
                actual_predictions = model1.predict(x_abs)
                expected_predictions = model2.predict(x_abs)
            else:
                actual_predictions = model1.predict(np.absolute(x))
                expected_predictions = model2.predict(np.absolute(x))
        else:
            actual_predictions = model1.predict(x)
            expected_predictions = model2.predict(x)
        testing.assert_array_almost_equal(actual_predictions, expected_predictions)
    elif isinstance(model1, TransformerModel) and isinstance(model2, TransformerModel):
        if abs:
            if isinstance(x, csr_matrix):
                x_abs = csr_matrix(
                    (
                        np.absolute(x.data),
                        x.indices.astype(np.int32),
                        x.indptr.astype(np.int32),
                    ),
                    shape=x.shape,
                )
                actual_predictions = model1.transform(x_abs)
                expected_predictions = model2.transform(x_abs)
            else:
                actual_predictions = model1.transform(np.absolute(x))
                expected_predictions = model2.transform(np.absolute(x))
        else:
            actual_predictions = model1.transform(x)
            expected_predictions = model2.transform(x)
        if isinstance(actual_predictions, csr_matrix) and isinstance(
            expected_predictions, csr_matrix
        ):
            assert_sparse_matrix_equal(actual_predictions, expected_predictions)
        else:
            testing.assert_array_almost_equal(actual_predictions, expected_predictions)


def run_test_transformed_data(
    model1: TransformerModel, model2: TransformerModel, x: Union[np.ndarray, csr_matrix]
) -> None:
    """
    Compares the transformed data of two models on the given data.

    Parameters
    ----------
    model1 : TransformerModel
        The first scikit-learn model with a transform method.
    model2 : TransformerModel
        The second scikit-learn model with a transform method.
    x : Union[np.ndarray, csr_matrix]
        The input samples.

    Raises
    ------
    AssertionError
        If the transformed data of the two models are not almost equal.
    """
    expected_transformed_data = model1.transform(x)
    actual_transformed_data = model2.transform(x)
    if isinstance(expected_transformed_data, csr_matrix) and isinstance(
        actual_transformed_data, csr_matrix
    ):
        assert_sparse_matrix_equal(expected_transformed_data, actual_transformed_data)
    else:
        testing.assert_array_almost_equal(
            expected_transformed_data, actual_transformed_data
        )


def _roundtrip_and_compare(
    fitted_model: FittableModel,
    x: Union[np.ndarray, csr_matrix],
    abs: bool,
    model_name: str,
) -> None:
    """
    Serialize `fitted_model` to JSON, deserialize it (in memory and through a file), and check
    that both copies predict/transform `x` exactly like the original. Assertion errors are
    prefixed with `model_name`, so a failure says which fit (e.g. dense or sparse) it was.
    """
    manager = SerializationManager(SklearnSerializer())
    serialized_model = manager.serialize(fitted_model, format_name="json")

    def compare(deserialized_model) -> None:
        if isinstance(fitted_model, PredictorModel):
            run_test_predictions(
                cast(PredictorModel, fitted_model),
                cast(PredictorModel, deserialized_model),
                x,
                abs,
            )
        elif isinstance(fitted_model, TransformerModel):
            run_test_transformed_data(
                cast(TransformerModel, fitted_model),
                cast(TransformerModel, deserialized_model),
                x,
            )

    model_file_path = f"./test/temp/{model_name}.json"
    os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
    try:
        compare(manager.deserialize(serialized_model, format_name="json"))

        with open(model_file_path, "w", encoding="utf-8") as f:
            f.write(serialized_model)
        with open(model_file_path, "r", encoding="utf-8") as f:
            serialized_model_from_file = f.read()
        compare(manager.deserialize(serialized_model_from_file, format_name="json"))
    except AssertionError as e:
        raise AssertionError(f"[{model_name}] {e}") from e
    finally:
        if os.path.exists(model_file_path):
            os.remove(model_file_path)


def run_test_model(
    model: FittableModel,
    x: Union[np.ndarray, csr_matrix],
    y: np.ndarray,
    x_sparse: Optional[Union[np.ndarray, csr_matrix]],
    y_sparse: Optional[np.ndarray],
    model_name: str,
    abs: bool = False,
) -> None:
    """
    Tests the serialization and deserialization of a scikit-learn model.

    The model is fitted on `x`/`y` and round-tripped. If `x_sparse` and `y_sparse` are both
    given, a separate copy is also fitted on them and round-tripped, compared on `x_sparse`.
    Each fit uses its own clone of `model` (which is left unfitted): fitting both on one
    instance would overwrite the dense fit, and sparse input changes what many estimators
    learn (e.g. neighbors estimators then use brute force instead of a search tree).

    Parameters
    ----------
    model : Union[BaseEstimator, ModelType]
        The scikit-learn model to test. Not modified.
    x : Union[np.ndarray, csr_matrix]
        The training input samples.
    y : np.ndarray
        The target values (class labels in classification, real numbers in regression).
    x_sparse : Optional[Union[np.ndarray, csr_matrix]]
        The sparse training input samples.
    y_sparse : np.ndarray or None
        The sparse target values. The sparse fit is only tested when this is given.
    model_name : str
        The name of the file to save the serialized model to.
    abs : bool, default=False
        Whether to take the absolute value of the input data before fitting the model.
    """
    x = ensure_correct_sparse_format(x)
    dense_model = fit_model(clone(model), x, y, abs)
    _roundtrip_and_compare(dense_model, x, abs, model_name)

    if x_sparse is not None and y_sparse is not None:
        x_sparse = ensure_correct_sparse_format(x_sparse)
        sparse_model = fit_model(clone(model), x_sparse, y_sparse, abs)
        _roundtrip_and_compare(sparse_model, x_sparse, abs, f"{model_name}_sparse")


def create_test_data(
    n_samples: int = 100, n_features: int = 5, random_state: int = 42
) -> tuple:
    """
    Creates test data for model fitting and testing.

    Parameters
    ----------
    n_samples : int, optional
        The number of samples to generate (default is 100).
    n_features : int, optional
        The number of features to generate (default is 5).
    random_state : int, optional
        The random state for reproducibility (default is 42).

    Returns
    -------
    tuple
        A tuple containing (X, y) where X is the feature matrix and y is the target vector.
    """
    np.random.seed(random_state)
    x = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, size=n_samples)
    return x, y


def run_test_label_binarizer(
    model: LabelBinarizer,
    y: np.ndarray,
    model_name: str,
) -> None:
    """
    Special test for LabelBinarizer: fit and transform only use y.
    """
    # Fit the model
    model.fit(y)
    transformed = model.transform(y)

    # Serialize and deserialize
    manager = SerializationManager(SklearnSerializer())
    serialized_model = manager.serialize(model, format_name="json")
    deserialized_model = manager.deserialize(serialized_model, format_name="json")

    # Test transform output
    transformed_deserialized = deserialized_model.transform(y)
    testing.assert_array_equal(transformed, transformed_deserialized)

    # File round-trip
    model_file_path = f"./test/temp/{model_name}.json"
    os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
    with open(model_file_path, "w", encoding="utf-8") as f:
        f.write(serialized_model)
    with open(model_file_path, "r", encoding="utf-8") as f:
        serialized_model_from_file = f.read()
    deserialized_model_from_file = manager.deserialize(
        serialized_model_from_file, format_name="json"
    )
    transformed_from_file = deserialized_model_from_file.transform(y)
    testing.assert_array_equal(transformed, transformed_from_file)
    os.remove(model_file_path)


def test_multilabelbinarizer_minimal():
    from sklearn.preprocessing import MultiLabelBinarizer
    from openmodels.serializers.sklearn.sklearn_serializer import SklearnSerializer
    from openmodels import SerializationManager
    import numpy as np

    # Integer labels
    y_int = [(1, 2), (3,)]
    mlb = MultiLabelBinarizer()
    transformed = mlb.fit_transform(y_int)
    classes = mlb.classes_.copy()

    manager = SerializationManager(SklearnSerializer())
    serialized = manager.serialize(mlb)
    mlb2 = manager.deserialize(serialized)

    np.testing.assert_array_equal(classes, mlb2.classes_)
    np.testing.assert_array_equal(transformed, mlb2.transform(y_int))

    # String labels
    y_str = [{"sci-fi", "thriller"}, {"comedy"}]
    mlb = MultiLabelBinarizer()
    transformed = mlb.fit_transform(y_str)
    classes = list(mlb.classes_)

    serialized = manager.serialize(mlb)
    mlb2 = manager.deserialize(serialized)

    assert list(mlb2.classes_) == classes
    np.testing.assert_array_equal(transformed, mlb2.transform(y_str))


def test_feature_hasher_serialization():
    # Example 1: Using dictionaries as input
    hasher_dict = FeatureHasher(n_features=10)
    D = [{"dog": 1, "cat": 2, "elephant": 4}, {"dog": 2, "run": 5}]
    transformed_dict = hasher_dict.transform(D).toarray()

    # Example 2: Using strings as input
    hasher_string = FeatureHasher(n_features=8, input_type="string")
    raw_X = [["dog", "cat", "snake"], ["snake", "dog"], ["cat", "bird"]]
    transformed_string = hasher_string.transform(raw_X).toarray()

    # Assertions for Example 1
    assert transformed_dict.shape == (2, 10), "Unexpected shape for dictionary input"
    assert not np.all(
        transformed_dict == 0
    ), "Transformation failed for dictionary input"

    # Assertions for Example 2
    assert transformed_string.shape == (3, 8), "Unexpected shape for string input"
    assert not np.all(transformed_string == 0), "Transformation failed for string input"


def test_generic_univariate_select_serialization():
    # Create the transformer
    transformer = GenericUnivariateSelect(
        score_func=f_classif, mode="percentile", param=50
    )

    # Serialize the transformer
    serializer = SklearnSerializer()
    serialized = serializer.serialize(transformer)

    # Deserialize the transformer
    deserialized = serializer.deserialize(serialized)

    # Check that the deserialized transformer works as expected
    assert deserialized.score_func == f_classif
    assert deserialized.mode == "percentile"
    assert deserialized.param == 50
