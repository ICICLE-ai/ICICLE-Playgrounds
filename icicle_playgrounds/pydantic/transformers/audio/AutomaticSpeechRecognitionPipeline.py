from typing import Any
import numpy as np
from pydantic import field_validator

from icicle_playgrounds.pydantic.transformers.PipelineIO import PipelineIO


class AutomaticSpeechRecognitionPipelineInput(PipelineIO):
    inputs: str | bytes | np.ndarray | dict[str, int | np.ndarray]
    return_timestamps: str | bool | None = None
    generate_kwargs: dict[str, Any] | None = None

    # Private Attributes
    _ndims = 1
    _dtypes = {np.float32, np.float64}
    _dict_keys = {frozenset({"sampling_rate", "raw"}), frozenset({"sampling_rate", "raw", "stride"})}

    @field_validator("inputs", mode="before")
    @classmethod
    def _prevalidate_inputs_value(
        cls, value: str | bytes | np.ndarray | dict[str, int | np.ndarray]
    ) -> str | bytes | np.ndarray | dict[str, int | np.ndarray]:
        if isinstance(value, str):
            if value.startswith("file:/"):
                # Input value is a file path, need to strip 'file:/' prefix
                return value.split("file:/")[1]
            elif value.startswith("base64"):
                # Input value should be a base64 encoded string. Need to decode to NumPy Array
                value = cls._base64_to_ndarray(encoded_str=value)
                return value
            else:
                raise ValueError("Invalid string input format")
        elif isinstance(value, list):
            # Input could be a list of floats, will convert to NumPy Array
            return np.array(value)
        elif isinstance(value, dict):
            # Input is a dictionary where a key could be a base64 encoded
            #   NumPy array or a list to convert to a NumPy array.
            if frozenset(value.keys()) not in cls._dict_keys:
                # Make sure the dictionary contains
                raise ValueError(
                    "Input dict must have keys 'sampling_rate' and 'raw' and optionally 'stride'."
                )
            array = value["raw"]
            if isinstance(array, str):
                # Array is a Base64 encoded string
                array = cls._base64_to_ndarray(encoded_str=array)
            elif isinstance(array, list):
                # Array is a python list
                array = np.array(array)
            else:
                raise ValueError(
                    "Invalid input 'raw' format, it needs to be a base64 encoded string or a python list"
                )
            # Array should be a numpy array now.
            value["raw"] = array
        return value

    @field_validator("inputs", mode="after")
    @classmethod
    def _validate_inputs_value(
        cls, value: str | bytes | np.ndarray | dict[str, int | np.ndarray]
    ) -> str | bytes | np.ndarray | dict[str, int | np.ndarray]:
        if value is None:
            raise ValueError("Inputs cannot be None")
        if isinstance(value, (str, bytes)):
            return value
        elif isinstance(value, np.ndarray):
            # Need to validate numpy array dimensions and dtype
            cls._validate_ndarray_input(value)
            return value
        elif isinstance(value, dict):
            # Keys have already been prevalidated,
            # we just need to validate the numpy array in 'raw', 'smapling_rate', and optionally 'stride' type.
            if not isinstance(value["sampling_rate"], int):
                raise ValueError("Sampling rate must be an integer")
            if "stride" in value.keys():
                if (not isinstance(value["stride"], tuple) and len(value["stride"]) != 2 and
                        not isinstance(value["stride"][0], int) and not isinstance(value["stride"][1], int)):
                    raise ValueError("Stride must be a tuple of integers")
            cls._validate_ndarray(value["raw"])
            return value
        else:
            raise ValueError("Invalid inputs format")

    @field_validator("return_timestamps", mode="after")
    @classmethod
    def _validate_return_timestamps_value(
        cls, value: str | bool | None
    ) -> str | bool | None:
        if value is None:
            return None
        if isinstance(value, str):
            if value.lower() not in {"char", "word"}:
                raise ValueError("return_timestamps string value must be 'char' or 'word'.")
            return value.lower()
        elif isinstance(value, bool):
            return value
        else:
            raise ValueError("return_timestamps must be a string or a boolean")

class AutomaticSpeechRecognitionPipelineOutput(PipelineIO):
    results: dict[str, str | list[dict[str, str | tuple[float, float]]]] = {}

    _dict_keys = {frozenset({"text"}),frozenset({"text", "chunks"}), frozenset({"text", "timestamp"})}

    @field_validator("results", mode="after")
    @classmethod
    def _validate_results_value(
        cls, value: dict[str, str | list[dict[str, str | tuple[float, float]]]]
    ) -> dict[str, str | list[dict[str, str | tuple[float, float]]]]:
        if not isinstance(value, dict):
            raise ValueError("Results must be a dictionary")
        if frozenset(value.keys()) not in cls._dict_keys:
            raise ValueError("Results must have key 'text' and optionally a 'chunks' key.")
        if not isinstance(value["text"], str):
            raise ValueError("Results 'text' value must be a string.")
        if "chunks" in value.keys():
            if not isinstance(value["chunks"], list):
                raise ValueError("Results 'chunks' value must be a list.")
            for item in value["chunks"]:
                if not isinstance(item, dict):
                    raise ValueError("Results 'chunks' items must be dictionaries.")
                if item.keys() not in cls._dict_keys:
                    raise ValueError("Results 'chunks' items must have keys 'text' and 'timestamp'.")
                if not isinstance(item["text"], str):
                    raise ValueError("Results 'chunks' item 'text' value must be a string.")
                if (not isinstance(item["timestamp"], tuple) and len(item["timestamp"]) != 2 and
                        not isinstance(item["timestamp"][0], float) and not isinstance(item["timestamp"][1], float)):
                    raise ValueError("Results 'chunks' item 'timestamp' value must be a tuple of floats.")
        return value