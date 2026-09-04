# Registers sklearn's own conftest as a plugin - required because some of the sklearn test
# modules reused wholesale under test/upstream/ (see test/upstream/cross_decomposition/ for the
# worked example) depend on fixtures sklearn defines there, e.g. `global_random_seed`, which is
# wired up via a pytest_generate_tests hook. pytest only allows pytest_plugins to be declared in
# a conftest.py at the true rootdir, which is why this lives here rather than under test/.
#
# Recipe for reusing another sklearn test module (e.g. sklearn/svm/tests/test_svm.py):
#   1. mkdir test/upstream/<package>/ with an empty __init__.py.
#   2. test/upstream/<package>/conftest.py: a module-scoped autouse fixture wrapping
#      openmodels.test_helpers.roundtrip_fit around the estimator classes that module's tests
#      exercise, e.g.:
#          @pytest.fixture(scope="module", autouse=True)
#          def _roundtrip_something():
#              with roundtrip_fit(EstimatorA, EstimatorB):
#                  yield
#   3. test/upstream/<package>/test_<module>_upstream.py, a one-liner:
#          from sklearn.<package>.tests.test_<module> import *  # noqa: F401,F403
#   4. Run it and skim the output. Not everything in an upstream test file is fair game as a
#      round-trip check (some test private helper functions unrelated to any estimator - that's
#      harmless bonus coverage) and not everything that fails is openmodels's fault (some fail
#      identically against a bare, unpatched estimator - verify with roundtrip_fit removed
#      before assuming a failure is a real serialization gap; see
#      test/test_estimator_conformance.py's KNOWN_ROUNDTRIP_XFAILS for the established pattern
#      of documenting which is which, with reasons).
pytest_plugins = ["sklearn.conftest"]
