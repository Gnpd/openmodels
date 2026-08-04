"""
Reuses scikit-learn's own Birch test suite unmodified. The autouse fixture in conftest.py
round-trips Birch through openmodels immediately after fit()/fit_transform()/fit_predict(), so
every assertion sklearn's own maintainers wrote here - including scenarios that call
partial_fit() after fit(), which is what actually exercises Birch's root_/dummy_leaf_ CF-tree
structure - doubles as an openmodels fidelity check for this estimator.
"""

from sklearn.cluster.tests.test_birch import *  # noqa: F401,F403
