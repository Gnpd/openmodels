"""
ColumnTransformer must work after loading: transform (numpy and pandas), get_feature_names_out,
set_output(transform="pandas"), slice and make_column_selector columns, and inside a Pipeline
fed a DataFrame (the common real-world setup).
"""

import json

import numpy as np
import pytest
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)

from openmodels import SerializationManager, SklearnSerializer

rng = np.random.RandomState(0)
X = rng.rand(30, 4)
X_CAT = np.column_stack([rng.rand(30), rng.randint(0, 3, 30)])


def _roundtrip(model, fmt="json"):
    manager = SerializationManager(SklearnSerializer())
    return manager.deserialize(
        manager.serialize(model, format_name=fmt), format_name=fmt
    )


def _dense(a):
    return a.toarray() if hasattr(a, "toarray") else np.asarray(a)


def _assert_same(loaded, model, data):
    np.testing.assert_allclose(
        _dense(loaded.transform(data)), _dense(model.transform(data))
    )
    assert list(loaded.get_feature_names_out()) == list(model.get_feature_names_out())
    assert sorted(loaded.named_transformers_) == sorted(model.named_transformers_)
    assert all(isinstance(s, slice) for s in loaded.output_indices_.values())
    assert loaded.output_indices_ == model.output_indices_


# ==== numpy input ====

NUMPY_CASES = {
    "passthrough": (
        lambda: ColumnTransformer(
            [("s", StandardScaler(), [0, 1])], remainder="passthrough"
        ),
        X,
    ),
    "two_transformers_drop": (
        lambda: ColumnTransformer(
            [("s", StandardScaler(), [0, 1]), ("m", MinMaxScaler(), [2])]
        ),
        X,
    ),
    "one_hot_sparse": (
        lambda: ColumnTransformer(
            [("s", StandardScaler(), [0]), ("o", OneHotEncoder(), [1])]
        ),
        X_CAT,
    ),
    "slice_columns_estimator_remainder": (
        lambda: ColumnTransformer(
            [("s", StandardScaler(), slice(0, 2))], remainder=MinMaxScaler()
        ),
        X,
    ),
}


@pytest.mark.parametrize("case", NUMPY_CASES)
def test_numpy_column_transformer_roundtrips(case):
    make, data = NUMPY_CASES[case]
    model = make().fit(data)
    _assert_same(_roundtrip(model), model, data)


def test_column_transformer_roundtrips_with_pickle():
    make, data = NUMPY_CASES["slice_columns_estimator_remainder"]
    model = make().fit(data)
    _assert_same(_roundtrip(model, "pickle"), model, data)


def test_file_written_by_earlier_versions_still_transforms_numpy():
    # 0.2.x didn't save _transformer_to_input_indices; numpy transform worked without it.
    make, data = NUMPY_CASES["two_transformers_drop"]
    model = make().fit(data)
    serializer = SklearnSerializer()
    serialized = json.loads(json.dumps(serializer.serialize(model)))
    del serialized["attributes"]["_transformer_to_input_indices"]
    loaded = serializer.deserialize(serialized)
    np.testing.assert_allclose(loaded.transform(data), model.transform(data))


# ==== slices anywhere ====


def test_slice_param_roundtrips():
    # Any slice used to crash on load: slice() takes no keyword arguments.
    model = FunctionTransformer(kw_args={"cols": slice(1, 3)})
    serializer = SklearnSerializer()
    assert serializer.convert_from_serializable(
        {"start": 1, "stop": 3, "step": None}, "slice"
    ) == slice(1, 3)
    assert _roundtrip(model).kw_args == {"cols": slice(1, 3)}


# ==== NumPy abstract types (used by make_column_selector dtype specs) ====


@pytest.mark.parametrize("t", [np.number, np.integer, np.floating, np.inexact])
def test_numpy_abstract_type_roundtrips(t):
    serializer = SklearnSerializer()
    written = json.loads(json.dumps(serializer.convert_to_serializable(t)))
    assert serializer.convert_from_serializable(written, "type") is t


# ==== pandas input ====


def _frame():
    pd = pytest.importorskip("pandas")
    return pd.DataFrame(
        {
            "a": rng.rand(30),
            "b": rng.rand(30),
            "c": rng.choice(["x", "y", "z"], 30),
            "d": rng.rand(30),
        }
    )


def test_pandas_named_columns_roundtrip():
    df = _frame()
    model = ColumnTransformer(
        [("num", StandardScaler(), ["a", "b"]), ("cat", OneHotEncoder(), ["c"])]
    ).fit(df)
    _assert_same(_roundtrip(model), model, df)


@pytest.mark.parametrize(
    "selector",
    [
        make_column_selector(dtype_include=np.number),
        make_column_selector(dtype_include="number"),
        make_column_selector(dtype_exclude=[object, "category"]),
        make_column_selector(pattern="^[ab]$"),
    ],
    ids=["np.number", "number_string", "exclude_list", "pattern"],
)
def test_pandas_make_column_selector_roundtrips(selector):
    df = _frame()
    model = ColumnTransformer([("sel", "passthrough", selector)]).fit(df)
    loaded = _roundtrip(model)
    _assert_same(loaded, model, df)
    restored = loaded.transformers[0][2]
    assert isinstance(restored, make_column_selector)
    assert restored(df) == selector(df)


def test_pandas_set_output_roundtrips():
    df = _frame()
    model = ColumnTransformer(
        [
            ("num", StandardScaler(), ["a", "b"]),
            ("cat", OneHotEncoder(sparse_output=False), ["c"]),
        ]
    ).fit(df)
    loaded = _roundtrip(model)
    out_model = model.set_output(transform="pandas").transform(df)
    out_loaded = loaded.set_output(transform="pandas").transform(df)
    assert list(out_loaded.columns) == list(out_model.columns)
    np.testing.assert_allclose(out_loaded.to_numpy(), out_model.to_numpy())


def test_pandas_pipeline_predicts_after_load():
    df = _frame()
    target = (df["a"] > 0.5).astype(int)
    model = Pipeline(
        [
            (
                "ct",
                ColumnTransformer(
                    [
                        ("num", StandardScaler(), ["a", "b"]),
                        ("cat", OneHotEncoder(), ["c"]),
                    ]
                ),
            ),
            ("clf", LogisticRegression()),
        ]
    ).fit(df, target)
    loaded = _roundtrip(model)
    np.testing.assert_array_equal(loaded.predict(df), model.predict(df))
    assert list(loaded[:-1].get_feature_names_out()) == list(
        model[:-1].get_feature_names_out()
    )
