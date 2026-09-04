import pytest
import numpy as np
from sklearn.utils.discovery import all_estimators
from sklearn.datasets import make_classification
from openmodels.test_helpers import run_test_model
from openmodels.serializers.sklearn.sklearn_serializer import NOT_SUPPORTED_ESTIMATORS
from test._estimator_construction import BASE_CLASSIFIER, CONSTRUCTOR_ARGS

# Get all classifier estimators, filtering out not supported classifiers
CLASSIFIERS = [cls for name, cls in all_estimators(type_filter="classifier")
        if name not in NOT_SUPPORTED_ESTIMATORS]

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

@pytest.mark.parametrize("Classifier", CLASSIFIERS)
def test_classifier(Classifier, data):
    x, y = data

    args = {}

    abs = False
    base_lr = BASE_CLASSIFIER

    if Classifier.__name__ in ["CategoricalNB", "ComplementNB", "MultinomialNB"]:
        abs = True
    elif Classifier.__name__ == "ClassifierChain":
        args.update(CONSTRUCTOR_ARGS["ClassifierChain"])
        y_multi = np.column_stack([(y == i).astype(int) for i in np.unique(y)])
        y = y_multi
    elif Classifier.__name__ in ["FixedThresholdClassifier", "TunedThresholdClassifierCV"]:
        args.update(CONSTRUCTOR_ARGS[Classifier.__name__])
        y_binary = (y == 0).astype(int)
        y = y_binary
    elif Classifier.__name__ in ["OneVsOneClassifier", "OutputCodeClassifier", "SelfTrainingClassifier"]:
        args.update(CONSTRUCTOR_ARGS[Classifier.__name__])
    elif Classifier.__name__ in ["MultiOutputClassifier", "OneVsRestClassifier"]:
        args.update(CONSTRUCTOR_ARGS[Classifier.__name__])
        y_multi = np.column_stack([(y == i).astype(int) for i in np.unique(y)])
        y = y_multi
    elif Classifier.__name__ == "StackingClassifier":
        # Data-dependent (one sub-estimator per class label) - can't be a static shared entry.
        args["estimators"] = [(str(name), base_lr) for name in np.unique(y)]
    elif Classifier.__name__ == "VotingClassifier":
        args.update(CONSTRUCTOR_ARGS["VotingClassifier"])

    classifier = Classifier(**args)
    
    # Run the test model
    run_test_model(classifier, x, y, None, None, f"{Classifier.__name__.lower()}.json", abs)
 
