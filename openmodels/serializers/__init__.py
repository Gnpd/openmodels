"""
Serializers module for the OpenModels library.

This module provides serializers for different types of machine learning models.
Currently, it includes a serializer for scikit-learn models.
"""

from .sklearn.sklearn_serializer import SklearnSerializer

__all__ = ["SklearnSerializer"]
