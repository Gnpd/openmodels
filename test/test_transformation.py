import pytest
import random
import numpy as np
from sklearn.utils.discovery import all_estimators
from sklearn.datasets import make_classification
from sklearn.feature_extraction import FeatureHasher
from openmodels.test_helpers import (
    run_test_model,
    run_test_label_binarizer,
    test_multilabelbinarizer_minimal,
    test_feature_hasher_serialization,
    test_generic_univariate_select_serialization,
)
from openmodels.serializers.sklearn.sklearn_serializer import NOT_SUPPORTED_ESTIMATORS
from test.test_regression import REGRESSORS
from test.test_classification import CLASSIFIERS

# Get all transformer estimators, filtering out not supported ones and those that are also regressors/classifiers
TRANSFORMERS = [
    cls for name, cls in all_estimators(type_filter="transformer")
    if name not in NOT_SUPPORTED_ESTIMATORS
    and name not in [reg.__name__ for reg in REGRESSORS]
    and name not in [clf.__name__ for clf in CLASSIFIERS]
]


@pytest.fixture(scope="module")
def data():
    # Generate dense data
    x, y = make_classification(
        n_samples=50,
        n_features=5,
        n_classes=3,
        n_informative=3,
        n_redundant=0,
        random_state=0,
        shuffle=False,
    )
    # Generate sparse data
    feature_hasher = FeatureHasher(n_features=5)
    features = [
        {
            "a": random.randint(0, 2),
            "b": random.randint(3, 5),
            "c": random.randint(6, 8),
        }
        for _ in range(0, 100)
    ]
    x_sparse = feature_hasher.transform(iter(features))
    return x, y, x_sparse

@pytest.mark.parametrize("Transformer", TRANSFORMERS)
def test_transformer(Transformer, data):
    x, y, x_sparse = data

    args = {}
    if Transformer.__name__ in ["AdditiveChi2Sampler", "MiniBatchNMF", "LatentDirichletAllocation", "NMF"]:
        x = np.abs(x)
    
    if Transformer.__name__ in ["CCA", "PLSCanonical"]:
        args["n_components"] = 1
    if Transformer.__name__ in ["FeatureHasher"]:
        y = None
        x = None
        x_sparse = None
    if Transformer.__name__ in ["DictVectorizer"]:
        y = None
        x = [{'foo': 1, 'bar': 2}, {'foo': 3, 'baz': 1}]
        x_sparse = None
    if Transformer.__name__ in ["GaussianRandomProjection"]:  
        rng = np.random.RandomState(42)
        x = rng.rand(25, 3000)
        args["random_state"] = rng
    if Transformer.__name__ in ["KernelCenterer"]:
        from sklearn.metrics.pairwise import pairwise_kernels
        X = [[ 1., -2.,  2.], [ -2.,  1.,  3.], [ 4.,  1., -2.]]
        x = pairwise_kernels(X, metric="linear")
    if Transformer.__name__ in ["PLSSVD"]:
        args["n_components"] = 1
    if Transformer.__name__ == "ColumnTransformer":
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        # Example input: 2 numeric columns, 1 categorical column
        x = np.array([
            [0.5, 1.0, 'A'],
            [1.5, 2.0, 'B'],
            [3.0, 3.5, 'A'],
            [2.5, 2.0, 'C']
        ], dtype=object)
        y = None
        x_sparse = None
        args["transformers"] = [
            ("num", StandardScaler(), [0, 1]),
            ("cat", OneHotEncoder(), [2])
    ]
    if Transformer.__name__ == "FeatureUnion":
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        # Example input: numeric data
        x = np.array([
            [0.5, 1.0],
            [1.5, 2.0],
            [3.0, 3.5],
            [2.5, 2.0]
        ])
        y = None
        x_sparse = None
        args["transformer_list"] = [
            ("scaler1", StandardScaler()),
            ("scaler2", MinMaxScaler())
        ]
    if Transformer.__name__ in ["SelectFromModel", "SequentialFeatureSelector", "RFE", "RFECV"]:
        from sklearn.linear_model import LogisticRegression
        args["estimator"] = LogisticRegression()
    if Transformer.__name__ == "SparseCoder":
        # SparseCoder requires a dictionary (components) for initialization
        # Let's create a random dictionary with shape (n_components, n_features)
        n_components = 5
        n_features = x.shape[1] if x is not None else 5
        rng = np.random.RandomState(42)
        dictionary = rng.rand(n_components, n_features)
        args["dictionary"] = dictionary
    if Transformer.__name__ == "HashingVectorizer":
        y = None
        x = ["This is a test document.", "Another document for testing.", "Yet another one."]
        x_sparse = None
    if Transformer.__name__ == "LocalOutlierFactor":
        # Set novelty=True to enable the predict method
        args["novelty"] = True
    if Transformer.__name__ == "SkewedChi2Sampler":
        # Ensure X satisfies the condition X >= -skewedness
        skewedness = 0.5  # Default value for skewedness
        args["skewedness"] = skewedness
        x = np.abs(x) + skewedness  # Make all values >= -skewedness
    if Transformer.__name__ == "SparseRandomProjection":
        # Explicitly set n_components to avoid the ValueError
        args["n_components"] = min(x.shape[1], 3)  # Set to a small value


    transformer = Transformer(**args)

    if Transformer.__name__ == "MultiLabelBinarizer":
        test_multilabelbinarizer_minimal()
        return
    if Transformer.__name__ in ("LabelBinarizer", "LabelEncoder"):
        # Ensure y is a 1D array of discrete labels
        _, y_int = np.unique(y, return_inverse=True)
        y_int = y_int.astype(int)
        # Only y is used for fit/transform; x is ignored
        run_test_label_binarizer(transformer, y_int, f"{Transformer.__name__.lower()}.json")
        return
    if Transformer.__name__ in ["FeatureHasher"]:
        test_feature_hasher_serialization()
        return
    if Transformer.__name__ == "GenericUnivariateSelect":
        test_generic_univariate_select_serialization()
        return


    run_test_model(transformer, x, y, x_sparse, None, f"{Transformer.__name__.lower()}.json")
