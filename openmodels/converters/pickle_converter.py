"""
Pickle converter for the OpenModels library.

This module provides a converter for serializing to and from pickle format.

.. warning::
    Unpickling can execute arbitrary code. Only deserialize pickle data from sources you
    trust - see SECURITY.md. Prefer the JSON format for data from untrusted sources.
"""

import pickle
from typing import Any, Dict

from openmodels.protocols import FormatConverter


class PickleConverter(FormatConverter):
    """
    Converter for pickle format.

    This class provides static methods to convert between dictionary
    representations and pickle byte strings.

    .. warning::
        ``deserialize_from_format`` calls ``pickle.loads()``, which can execute arbitrary
        code as part of deserialization. Only use this converter with pickle data from
        sources you trust.
    """

    is_binary = True

    @staticmethod
    def serialize_to_format(data: Dict[str, Any]) -> bytes:
        """
        Convert a dictionary to a pickle byte string.

        Parameters
        ----------
        data : Dict[str, Any]
            The dictionary to convert.

        Returns
        -------
        bytes
            The pickle byte string representation of the data.
        """
        return pickle.dumps(data)

    @staticmethod
    def deserialize_from_format(formatted_data: bytes) -> Dict[str, Any]:
        """
        Convert a pickle byte string to a dictionary.

        Parameters
        ----------
        formatted_data : bytes
            The pickle byte string to convert.

        Returns
        -------
        Dict[str, Any]
            The dictionary representation of the pickle data.

        Raises
        ------
        pickle.UnpicklingError
            If the input bytes cannot be unpickled.

        Warns
        -----
        Unpickling can execute arbitrary code. Only call this with pickle data from a
        source you trust.
        """
        return pickle.loads(formatted_data)
