import pytest
from sklearn.utils.discovery import all_estimators
from sklearn.datasets import make_classification
from openmodels.test_helpers import run_test_model
from openmodels.serializers.sklearn.sklearn_serializer import NOT_SUPPORTED_ESTIMATORS
from test.test_classification import CLASSIFIERS
from test.test_clustering import CLUSTERS
from test.test_regression import REGRESSORS
from test.test_transformation import TRANSFORMERS

# Get all other estimators, filtering out not supported
OTHERS = [cls for name, cls in all_estimators()
        if cls not in CLASSIFIERS + CLUSTERS + REGRESSORS + TRANSFORMERS
        and name not in NOT_SUPPORTED_ESTIMATORS
        ]

# Define constants
N_SAMPLES = 50
N_FEATURES = 3
N_CLASSES = 3
N_INFORMATIVE = 3
N_REDUNDANT = 0
RANDOM_STATE = 0
N_ESTIMATORS = 10
MAX_DEPTH = 5
LEARNING_RATE = 1.0
SOLVER = "lbfgs"
ALPHA = 1e-5
HIDDEN_LAYER_SIZES = (5, 2)
N_COMPONENTS = 2


@pytest.fixture(scope="module")
def data():
    # Generate test data
    x, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_classes=N_CLASSES,
        n_informative=N_INFORMATIVE,
        n_redundant=N_REDUNDANT,
        random_state=RANDOM_STATE,
        shuffle=False,
    )

    return x, y

@pytest.mark.parametrize("Others", OTHERS)
def test_others(Others, data):
    x, y = data

    args = {}
    # Special handling for text vectorizers
    if Others.__name__ in ["CountVectorizer", "TfidfVectorizer"]:
        # Provide a list of strings as input data
        x = [
            "this is a test document",
            "another test document",
            "text data for vectorizer",
            "machine learning is fun",
            "openmodels serialization test"
        ]
        y = None  # y is not used for vectorizers
    if Others.__name__ == "FrozenEstimator":
        from sklearn.linear_model import LogisticRegression
        base_estimator = LogisticRegression()
        base_estimator.fit(x, y)
        args["estimator"] = base_estimator
    if Others.__name__ == "GridSearchCV":
        from sklearn.linear_model import LogisticRegression
        # Define a simple parameter grid
        param_grid = {"C": [0.1, 1.0]}
        # Use a simple base estimator
        base_estimator = LogisticRegression()
        args["estimator"] = base_estimator
        args["param_grid"] = param_grid
    if Others.__name__ == "Pipeline":
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        # Example pipeline: scaler + classifier
        args["steps"] = [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression())  # The last step must be a predictor!
    ]
    if Others.__name__ == "RandomizedSearchCV":
        from sklearn.linear_model import LogisticRegression
        from scipy.stats import uniform
        # Define a simple parameter distribution
        param_distributions = {"C": uniform(0.1, 1.0)}
        # Use a simple base estimator
        base_estimator = LogisticRegression()
        args["estimator"] = base_estimator
        args["param_distributions"] = param_distributions

    others = Others(**args)

    # Run the test model
    run_test_model(others, x, y, None, None, f"{Others.__name__.lower()}.json")
 
