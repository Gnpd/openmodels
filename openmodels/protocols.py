"""
Protocol definitions for the OpenModels library.

This module defines the protocols (interfaces) that model serializers and
format converters must implement to be compatible with the SerializationManager.
"""

from abc import abstractmethod
from typing import Any, ClassVar, Dict, Protocol


class ModelSerializer(Protocol):
    """Protocol for model serializers."""

    @abstractmethod
    def serialize(self, model: Any) -> Dict[str, Any]:
        """
        Serialize a model to a dictionary representation.

        Parameters
        ----------
        model : Any
            The model to serialize.

        Returns
        -------
        Dict[str, Any]
            A dictionary representation of the model.
        """

    @abstractmethod
    def deserialize(self, data: Dict[str, Any]) -> Any:
        """
        Deserialize a model from a dictionary representation.

        Parameters
        ----------
        data : Dict[str, Any]
            The dictionary representation of the model.

        Returns
        -------
        Any
            The deserialized model.
        """


class FormatConverter(Protocol):
    """Protocol for format converters."""

    #: Whether `serialize_to_format` returns `bytes` (and a file for this format must
    #: therefore be opened in binary mode) rather than `str`. `SerializationManager.load`
    #: reads this to pick the right file mode instead of hardcoding format names.
    is_binary: ClassVar[bool]

    @staticmethod
    @abstractmethod
    def serialize_to_format(data: Dict[str, Any]) -> Any:
        """
        Convert a dictionary to a specific format.

        Parameters
        ----------
        data : Dict[str, Any]
            The dictionary to convert.

        Returns
        -------
        Any
            The data in the specific format.
        """

    @staticmethod
    @abstractmethod
    def deserialize_from_format(formatted_data: Any) -> Dict[str, Any]:
        """
        Convert data from a specific format to a dictionary.

        Parameters
        ----------
        formatted_data : Any
            The data in the specific format.

        Returns
        -------
        Dict[str, Any]
            The dictionary representation of the data.
        """
