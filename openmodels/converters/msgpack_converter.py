"""
MessagePack converter for the OpenModels library.

This module provides a converter for serializing to and from MessagePack format - a
binary format with the same data model as JSON (no code execution risk on load), useful when
JSON's text encoding is too slow/large for a given model. Requires the optional ``msgpack``
package (``pip install openmodels[msgpack]``).
"""

from typing import Any, Dict

from openmodels.protocols import FormatConverter

_INSTALL_HINT = (
    "The 'msgpack' package is required for format_name=\"msgpack\". "
    "Install it with: pip install openmodels[msgpack]"
)


class MsgpackConverter(FormatConverter):
    """
    Converter for MessagePack format.

    This class provides static methods to convert between dictionary
    representations and MessagePack byte strings.
    """

    is_binary = True

    @staticmethod
    def serialize_to_format(data: Dict[str, Any]) -> bytes:
        """
        Convert a dictionary to a MessagePack byte string.

        Parameters
        ----------
        data : Dict[str, Any]
            The dictionary to convert.

        Returns
        -------
        bytes
            The MessagePack byte string representation of the data.
        """
        try:
            import msgpack
        except ImportError as e:
            raise ImportError(_INSTALL_HINT) from e

        if not isinstance(data, dict):
            raise TypeError("Data must be a dictionary.")
        return msgpack.packb(data, use_bin_type=True)

    @staticmethod
    def deserialize_from_format(formatted_data: bytes) -> Dict[str, Any]:
        """
        Convert a MessagePack byte string to a dictionary.

        Parameters
        ----------
        formatted_data : bytes
            The MessagePack byte string to convert.

        Returns
        -------
        Dict[str, Any]
            The dictionary representation of the MessagePack data.
        """
        try:
            import msgpack
        except ImportError as e:
            raise ImportError(_INSTALL_HINT) from e

        return msgpack.unpackb(formatted_data, raw=False)
