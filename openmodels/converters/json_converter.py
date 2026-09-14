"""
JSON converter for the OpenModels library.

This module provides a converter for serializing to and from JSON format.
"""

import json
from typing import Any, Dict

from openmodels.protocols import FormatConverter


class JSONConverter(FormatConverter):
    """
    Converter for JSON format.

    This class provides static methods to convert between dictionary
    representations and JSON strings.
    """

    is_binary = False

    @staticmethod
    def serialize_to_format(data: Dict[str, Any]) -> str:
        """
        Convert a dictionary to a JSON string.

        Parameters
        ----------
        data : Dict[str, Any]
            The dictionary to convert.

        Returns
        -------
        str
            The JSON string representation of the data.
        """

        if not isinstance(data, dict):
            raise TypeError("Data must be a dictionary.")
        return json.dumps(data)

    @staticmethod
    def deserialize_from_format(formatted_data: str) -> Dict[str, Any]:
        """
        Convert a JSON string to a dictionary.

        Parameters
        ----------
        formatted_data : str
            The JSON string to convert.

        Returns
        -------
        Dict[str, Any]
            The dictionary representation of the JSON data.

        Raises
        ------
        json.JSONDecodeError
            If the input string is not valid JSON.
        """
        return json.loads(formatted_data)
