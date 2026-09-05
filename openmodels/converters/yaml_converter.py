"""
YAML converter for the OpenModels library.

This module provides a converter for serializing to and from YAML format - useful when a
human wants to read, hand-edit, or diff a model's serialized ``metadata`` cleanly. Requires
the optional ``PyYAML`` package (``pip install openmodels[yaml]``).

.. warning::
    This converter only ever calls ``yaml.safe_load``/``yaml.safe_dump``, never ``yaml.load``/
    ``yaml.dump`` with the default (unsafe) ``Loader``/``Dumper``, which support a
    ``!!python/object`` tag that can construct arbitrary Python objects on load - YAML's
    equivalent of pickle's arbitrary-code-execution risk.
"""

from typing import Any, Dict

from openmodels.protocols import FormatConverter

_INSTALL_HINT = (
    "The 'PyYAML' package is required for format_name=\"yaml\". "
    "Install it with: pip install openmodels[yaml]"
)


class YAMLConverter(FormatConverter):
    """
    Converter for YAML format.

    This class provides static methods to convert between dictionary
    representations and YAML strings, using only the safe loader/dumper.
    """

    is_binary = False

    @staticmethod
    def serialize_to_format(data: Dict[str, Any]) -> str:
        """
        Convert a dictionary to a YAML string.

        Parameters
        ----------
        data : Dict[str, Any]
            The dictionary to convert.

        Returns
        -------
        str
            The YAML string representation of the data.
        """
        try:
            import yaml
        except ImportError as e:
            raise ImportError(_INSTALL_HINT) from e

        if not isinstance(data, dict):
            raise TypeError("Data must be a dictionary.")
        # sort_keys=False keeps the natural field order (estimator_class, params, ...,
        # metadata) instead of alphabetizing it - readability is the whole point of this format.
        return yaml.safe_dump(data, sort_keys=False)

    @staticmethod
    def deserialize_from_format(formatted_data: str) -> Dict[str, Any]:
        """
        Convert a YAML string to a dictionary.

        Parameters
        ----------
        formatted_data : str
            The YAML string to convert.

        Returns
        -------
        Dict[str, Any]
            The dictionary representation of the YAML data.
        """
        try:
            import yaml
        except ImportError as e:
            raise ImportError(_INSTALL_HINT) from e

        return yaml.safe_load(formatted_data)
