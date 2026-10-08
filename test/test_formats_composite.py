"""
Composite estimators must round-trip in every format, and also when a serialize() result is
deserialized directly. Pickle and in-memory dicts keep tuples in the type maps (e.g. each
Pipeline step's type is a tuple), while JSON-like formats turn them into lists.
"""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from openmodels import SerializationManager, SklearnSerializer

X = np.random.RandomState(0).rand(60, 4)
y = (X[:, 0] + X[:, 1] > 1).astype(int)

MODELS = {
    "pipeline": lambda: Pipeline(
        [("scale", StandardScaler()), ("clf", LogisticRegression())]
    ),
    "feature_union": lambda: Pipeline(
        [
            (
                "union",
                FeatureUnion([("std", StandardScaler()), ("mm", MinMaxScaler())]),
            ),
            ("clf", LogisticRegression()),
        ]
    ),
    "column_transformer_in_pipeline": lambda: Pipeline(
        [
            (
                "ct",
                ColumnTransformer(
                    [("scale", StandardScaler(), [0, 1])], remainder="passthrough"
                ),
            ),
            ("clf", LogisticRegression()),
        ]
    ),
    "column_transformer": lambda: ColumnTransformer(
        [("scale", StandardScaler(), [0, 1]), ("mm", MinMaxScaler(), [2, 3])]
    ),
    "voting": lambda: VotingClassifier(
        [("lr", LogisticRegression()), ("dt", DecisionTreeClassifier(random_state=0))]
    ),
    "stacking": lambda: StackingClassifier([("lr", LogisticRegression())], cv=2),
}

FORMATS = ["json", "pickle", "msgpack", "yaml"]


def _output(model):
    return model.predict(X) if hasattr(model, "predict") else model.transform(X)


def _assert_fully_deserialized(loaded, original):
    np.testing.assert_array_equal(_output(loaded), _output(original))
    # No sub-estimator may be left as a raw serialized dict, wherever it sits in the params
    # (e.g. Voting/Stacking `estimators`, whose predict() doesn't use them).
    for name, value in loaded.get_params(deep=True).items():
        assert not (isinstance(value, dict) and "estimator_class" in value), name
        if isinstance(value, list):
            for item in value:
                parts = item if isinstance(item, (list, tuple)) else [item]
                assert not any(
                    isinstance(p, dict) and "estimator_class" in p for p in parts
                ), name
    clone(loaded)


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("model_name", MODELS)
def test_composite_roundtrip(model_name, fmt):
    if fmt in ("msgpack", "yaml"):
        pytest.importorskip(fmt)
    model = MODELS[model_name]().fit(X, y)
    manager = SerializationManager(SklearnSerializer())
    loaded = manager.deserialize(
        manager.serialize(model, format_name=fmt), format_name=fmt
    )
    _assert_fully_deserialized(loaded, model)


@pytest.mark.parametrize("model_name", MODELS)
def test_composite_roundtrip_in_memory(model_name):
    model = MODELS[model_name]().fit(X, y)
    serializer = SklearnSerializer()
    _assert_fully_deserialized(
        serializer.deserialize(serializer.serialize(model)), model
    )


def test_pipeline_pickle_file_roundtrip(tmp_path):
    model = MODELS["pipeline"]().fit(X, y)
    manager = SerializationManager(SklearnSerializer())
    path = tmp_path / "pipeline.pkl"
    manager.save(model, path, format_name="pickle")
    _assert_fully_deserialized(manager.load(path, format_name="pickle"), model)
