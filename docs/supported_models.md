# Supported Models

OpenModels supports **every** scikit-learn estimator, out of the box. The library has been
tested against scikit-learn versions **1.6.1**, **1.7.2**, **1.8.0**, and **1.9.0**.

## scikit-learn

All scikit-learn estimators are supported, including:

### Regressors

All scikit-learn regressors are supported, including:

- `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`
- `SVR`, `NuSVR`
- `KNeighborsRegressor`, `RadiusNeighborsRegressor`
- `GaussianProcessRegressor`
- `GradientBoostingRegressor`, `HistGradientBoostingRegressor`
- `PLSRegression`, `CCA`, `PLSCanonical`
- `IsotonicRegression`, `TransformedTargetRegressor`
- `PoissonRegressor`, `GammaRegressor`, `TweedieRegressor`
- And many more...

### Classifiers

All scikit-learn classifiers are supported, including:

- `LogisticRegression`, `RidgeClassifier`, `RidgeClassifierCV`
- `SVC`, `NuSVC`
- `KNeighborsClassifier`, `RadiusNeighborsClassifier`
- `GradientBoostingClassifier`, `HistGradientBoostingClassifier`
- `MLPClassifier`
- `StackingClassifier`
- `TunedThresholdClassifierCV`
- `DummyClassifier`
- And many more...

### Clustering

All scikit-learn clustering estimators are supported, including:

- `KMeans`, `MiniBatchKMeans`
- `BisectingKMeans`
- `Birch`
- And many more...

### Transformers

All scikit-learn transformers are supported, including:

- `PCA`, `KernelPCA`
- `OneHotEncoder`, `OrdinalEncoder`, `LabelBinarizer`
- `ColumnTransformer`
- `SimpleImputer`, `KNNImputer`
- `PowerTransformer`, `PolynomialFeatures`
- `TfidfVectorizer`
- `TargetEncoder`, `KBinsDiscretizer`
- `PatchExtractor` (expects an `(n_images, height, width[, n_channels])` image-batch array, not
  standard 2D tabular data)
- And many more...

### Other Estimators

All other scikit-learn estimators are supported, including:

- `IsolationForest`
- `OneClassSVM`
- `NearestNeighbors`
- `LocalOutlierFactor` (requires `novelty=True` for `predict()` - a scikit-learn API
  restriction, not an openmodels one; the default `novelty=False` mode is `fit_predict()`-only
  and has no `predict()` to round-trip)
- And many more...

Every scikit-learn estimator discoverable via `SklearnSerializer.all_estimators()` is supported -
see {doc}`format` for the details of what gets serialized.

## Custom & Third-Party Estimators

OpenModels supports custom estimators that follow scikit-learn's API via the
`custom_estimators` parameter in `SklearnSerializer`.

### chemotools

[chemotools](https://github.com/paucablop/chemotools) (>= 0.2.2) is a scikit-learn compatible
library for chemometrics preprocessing. It is fully supported via its built-in
`all_estimators` discovery function:

```python
from openmodels import SerializationManager, SklearnSerializer
from chemotools.utils.discovery import all_estimators
from chemotools.derivative import SavitzkyGolay
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import make_pipeline

# Build and fit a pipeline with chemotools + sklearn
pipeline = make_pipeline(
    SavitzkyGolay(window_size=3, polynomial_order=1, derivate_order=1),
    PLSRegression(n_components=2),
)
pipeline.fit(X_train, y_train)

# Serialize with custom estimators
serializer = SklearnSerializer(custom_estimators=all_estimators)
manager = SerializationManager(serializer)

serialized = manager.serialize(pipeline)
restored = manager.deserialize(serialized)
```

### Other Third-Party Packages

You can pass any compatible `all_estimators` function, list, or dictionary to
`SklearnSerializer(custom_estimators=...)` to extend support for your own estimators:

```python
manager = SerializationManager(
    SklearnSerializer(custom_estimators=my_custom_estimators)
)
```

If you maintain a scikit-learn compatible package and would like official support,
please [open an issue](https://github.com/Gnpd/openmodels/issues).

## Serialization Formats

OpenModels ships with two built-in format converters:

| Format | Converter | Human-Readable |
|---|---|---|
| JSON | `JSONConverter` | Yes |
| Pickle | `PickleConverter` | No |

Custom formats can be added by implementing the `FormatConverter` protocol and
registering with `FormatRegistry`. See the {doc}`getting_started` guide for details.
