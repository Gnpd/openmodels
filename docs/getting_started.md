# Getting Started

## Installation

```bash
pip install openmodels
```

## Quick Start

```python
from openmodels import SerializationManager, SklearnSerializer
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification

# Create and train a scikit-learn model
X, _ = make_classification(
    n_samples=1000, n_features=4, n_informative=2,
    n_redundant=0, random_state=0, shuffle=False,
)
model = PCA(n_components=2, random_state=0)
model.fit(X)

# Create a SerializationManager
manager = SerializationManager(SklearnSerializer())

# Serialize the model (default format is JSON)
serialized_model = manager.serialize(model)

# Deserialize the model
deserialized_model = manager.deserialize(serialized_model)

# Use the deserialized model
transformed_data = deserialized_model.transform(X[:5])
print(transformed_data)
```

## Saving and Loading Models

OpenModels provides high-level `save` and `load` methods for convenient file I/O:

```python
# Serialize and save a model to a file in JSON format
manager.save(model, "model.json", format_name="json")

# Load and deserialize a model from a file
loaded_model = manager.load("model.json", format_name="json")
```
