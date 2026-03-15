# Welcome to OpenModels

**OpenModels** is a flexible and extensible library for serializing and deserializing machine learning models.

It supports any serialization format through a plugin-based architecture, providing a safe and transparent solution for exporting and sharing predictive models. Currently, OpenModels offers built-in compatibility for **scikit-learn** estimators.

---

## Key Features

- **Format Agnostic** — Supports any serialization format through a plugin-based system.
- **Extensible** — Easily add support for new model types and serialization formats.
- **Safe** — Provides alternatives to potentially unsafe serialization methods like Pickle.
- **Transparent** — Supports human-readable formats for easy inspection of serialized models.

## Get Started

Install OpenModels with pip:

```bash
pip install openmodels
```

Then serialize your first model:

```python
from openmodels import SerializationManager, SklearnSerializer
from sklearn.linear_model import LogisticRegression

manager = SerializationManager(SklearnSerializer())
serialized = manager.serialize(model)
```

Check out the {doc}`getting_started` guide for a full walkthrough.

```{toctree}
:maxdepth: 2
:caption: Contents
:hidden:

getting_started
supported_models
api
```