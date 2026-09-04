"""
Reuses scikit-learn's own cross_decomposition test suite unmodified. The autouse fixture in
conftest.py round-trips PLSRegression/PLSCanonical/CCA/PLSSVD through openmodels immediately
after fit(), so every assertion sklearn's own maintainers wrote here doubles as an openmodels
fidelity check for this estimator family.
"""

from sklearn.cross_decomposition.tests.test_pls import *  # noqa: F401,F403
