"""
Core API module for the OpenModels library.

This module contains the SerializationManager class, which is the main interface
for serializing and deserializing machine learning models using various formats.
"""

from typing import Any, Union
from pathlib import Path
from .protocols import ModelSerializer
from .format_registry import FormatRegistry
from .exceptions import SerializationError, DeserializationError


class SerializationManager:
    """
    Manage the serialization and deserialization of machine learning models.

    This class coordinates the process of converting models to various formats
    and back, using specified model serializers and format converters.

    Attributes
    ----------
    model_serializer : ModelSerializer
        The serializer used to convert models to and from dictionary representations.
    """

    def __init__(self, model_serializer: ModelSerializer):
        """
        Initialize the SerializationManager with a model serializer.

        Parameters
        ----------
        model_serializer : ModelSerializer
            The serializer to use for converting models to and from dictionary representations.
        """
        self.model_serializer = model_serializer

    def serialize(self, model: Any, format_name: str = "json") -> Any:
        """
        Serialize a model to the specified format.

        Parameters
        ----------
        model : Any
            The machine learning model to serialize.
        format_name : str, optional
            The target format (default is "json").

        Returns
        -------
        Any
            The serialized model in the specified format.

        Raises
        ------
        SerializationError
            If the model serializer doesn't return a dictionary or if there's an error during
            serialization.
        UnsupportedFormatError
            If the specified format is not supported.

        Examples
        --------
        >>> manager = SerializationManager(SklearnSerializer())
        >>> model = LogisticRegression()
        >>> serialized_model = manager.serialize(model, format_name="json")
        """
        converter = FormatRegistry.get_converter(format_name)
        serialized_dict = self.model_serializer.serialize(model)
        if not isinstance(serialized_dict, dict):
            raise SerializationError(
                f"Model serializer must return a dict, got {type(serialized_dict)}"
            )
        return converter.serialize_to_format(serialized_dict)

    def deserialize(self, serialized_model: Any, format_name: str = "json") -> Any:
        """
        Deserialize a model from the specified format.

        Parameters
        ----------
        serialized_model : Any
            The serialized model data.
        format_name : str, optional
            The format of the serialized data (default is "json").

        Returns
        -------
        Any
            The deserialized machine learning model.

        Raises
        ------
        DeserializationError
            If the format converter doesn't return a dictionary or if there's an error during
            deserialization.
        UnsupportedFormatError
            If the specified format_name is not supported.

        Examples
        --------
        >>> manager = SerializationManager(SklearnSerializer())
        >>> deserialized_model = manager.deserialize(serialized_model, format_name="json")
        >>> predictions = deserialized_model.predict(X_test)
        """
        converter = FormatRegistry.get_converter(format_name)
        try:
            deserialized_dict = converter.deserialize_from_format(serialized_model)
        except Exception as e:
            raise DeserializationError(f"Error during deserialization: {e}")
        if not isinstance(deserialized_dict, dict):
            raise DeserializationError(
                f"Format converter must return a dict, got {type(deserialized_dict)}"
            )
        return self.model_serializer.deserialize(deserialized_dict)

    def save(
        self,
        model: Any,
        file_path: Union[str, Path, None] = None,
        format_name: str = "json",
    ) -> None:
        """
        Save a model to a file in the specified format.

        Parameters
        ----------
        model : Any
            The machine learning model to serialize and save.
        file_path : Union[str, Path, None]
            The path to the file where the model will be saved. If None, the model will be saved to
            a default filename of "model.{ext}", where "ext" is determined by the format (e.g., "json" or "pkl").
        format_name : str, optional
            The target format (default is "json").

        Raises
        ------
        SerializationError
            If there's an error during serialization or file I/O.
        UnsupportedFormatError
            If the specified format is not supported.

        Examples
        --------
        >>> manager = SerializationManager(SklearnSerializer())
        >>> model = LogisticRegression()
        >>> manager.save(model, "model.json", format_name="json")
        """
        serialized_data = self.serialize(model, format_name)

        # Determine file extension
        ext = format_name
        if format_name == "pickle":
            ext = "pkl"
        else:
            ext = format_name

        if file_path is None:
            file_path = f"model.{ext}"
        file_path = Path(file_path)

        try:
            # Determine write mode based on type of serialized_data
            if isinstance(serialized_data, bytes):
                with open(file_path, "wb") as f:
                    f.write(serialized_data)
            else:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(serialized_data)

        except Exception as e:
            raise SerializationError(f"Failed to save model: {e}")

    def load(self, file_path: Union[str, Path], format_name: str = "json") -> Any:
        """
        Load and deserialize a model from a file.

        Parameters
        ----------
        file_path : Union[str, Path]
            The path to the file containing the serialized model.
        format_name : str, optional
            The format of the serialized data (default is "json").

        Returns
        -------
        Any
            The deserialized machine learning model.

        Raises
        ------
        DeserializationError
            If deserialization or file I/O fails.
        UnsupportedFormatError
            If the specified format is not supported.

        Examples
        --------
        >>> manager = SerializationManager(SklearnSerializer())
        >>> loaded_model = manager.load("model.json", format_name="json")
        """
        file_path = Path(file_path)
        # Determine read mode based on format
        try:
            serialized_data: Union[str, bytes]
            if format_name == "pickle":
                with open(file_path, "rb") as f:
                    serialized_data = f.read()
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    serialized_data = f.read()
        except Exception as e:
            raise DeserializationError(f"Failed to load model: {e}")

        return self.deserialize(serialized_data, format_name)
