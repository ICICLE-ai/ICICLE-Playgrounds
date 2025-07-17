from enum import StrEnum

import numpy as np
from pydantic import ConfigDict, field_validator

from icicle_playgrounds.pydantic.transformers.PipelineIO import PipelineIO


class FunctionToApply(StrEnum):
    softmax = "softmax"
    sigmoid = "sigmoid"
    none = "none"



examples = {
    "1": {
        "inputs": "base64:eyJpbWFnZSI6Imh0dHBzOi8vd3d3Lm9tYWlsLmNvbS9tb2RlbC90ZXh0L3N0cmluZy9hZGRyZXNzL3N0cmluZy5wbmcifQ==",
        "top_k": 3,
        "function_to_apply": "softmax",
    },
    "2": {
        "inputs": "base64-compressed:eyJpbWFnZSI6Imh0dHBzOi8vd3d3Lm9tYWlsLmNvbS9tb2RlbC90ZXh0L3N0cmluZy9hZGRyZXNzL3N0cmluZy5wbmcifQ==",
        "top_k": 3,
        "function_to_apply": "sigmoid",
    },
    "3": {"inputs": "file:/path/to/audio.wav", "top_k": 3, "function_to_apply": "none"},
    "4": {"inputs": {"sampling_rate": 16000, "raw": b"audio_bytes"}, "top_k": 3},
    "5": {
        "inputs": {
            "sampling_rate": 16000,
            "array": "base64:eyJpbWFnZSI6Imh0dHBzOi8vd3d3Lm9tYWlsLmNvbS9tb2RlbC90ZXh0L3N0cmluZy9hZGRyZXNzL3N0cmluZy5wbmcifQ==",
        },
        "top_k": 3,
    },
    "6": {
        "inputs": {
            "sampling_rate": 16000,
            "raw": "base64:eyJpbWFnZSI6Imh0dHBzOi8vd3d3Lm9tYWlsLmNvbS9tb2RlbC90ZXh0L3N0cmluZy9hZGRyZXNzL3N0cmluZy5wbmcifQ==",
        },
        "top_k": None,
    },
    "7": {
        "inputs": {"sampling_rate": 16000, "array": [0.1, 0.2, 0.3]},
        "top_k": None,
    },
}


class AudioClassificationPipelineInput(PipelineIO):
    inputs: str | bytes | np.ndarray | dict[str, int | np.ndarray]
    top_k: int | None = None
    function_to_apply: FunctionToApply = FunctionToApply.softmax

    # Private Attributes
    _ndims = 1
    _dtypes = {np.float32, np.float64}
    _dict_keys = {frozenset({"sampling_rate", "raw"}), frozenset({"sampling_rate", "array"})}


    @field_validator("inputs", mode="before")
    @classmethod
    def _prevalidate_inputs_value(
        cls, value: str | bytes | np.ndarray | dict[str, int | np.ndarray]
    ) -> str | bytes | np.ndarray | dict[str, int | np.ndarray]:
        if isinstance(value, str):
            if value.startswith("file:/"):
                # Input value is a file path, need to strip 'file:/' prefix
                value = value.split("file:/")[1]
            elif value.startswith("base64"):
                # Input value should be a base64 encoded string. Need to decode to NumPy Array
                value = cls._base64_to_ndarray(encoded_str=value)
            else:
                raise ValueError("Invalid string input format")
        elif isinstance(value, list):
            # Input could be a list of floats, will convert to NumPy Array
            value = np.array(value)
        elif isinstance(value, dict):
            # Input is a dictionary where a key could be a base64 encoded
            # NumPy array or a list to convert to a NumPy array.
            if frozenset(value.keys()) not in cls._dict_keys:
                # Make sure the dictionary contains
                raise ValueError(
                    "Input dict must have keys 'sampling_rate' and 'raw' or 'sampling_rate' and 'array'"
                )
            array = value["array" if "array" in value else "raw"]
            if isinstance(array, str):
                # Array is a Base64 encoded string
                array = cls._base64_to_ndarray(encoded_str=array)
            elif isinstance(array, list):
                # Array is a python list
                array = np.array(array)
            else:
                raise ValueError("Invalid input 'array' or 'raw' format, it needs to be a base64 encoded string or a python list")
            # Array should be a numpy array now.
            value["array" if "array" in value else "raw"] = array
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
            # we just need to validate the numpy array in 'array' or 'raw' and 'smapling_rate' type.
            if not isinstance(value["sampling_rate"], int):
                raise ValueError("Sampling rate must be an integer")
            cls._validate_ndarray(value["array" if "array" in value else "raw"])
            return value
        else:
            raise ValueError("Invalid inputs format")


class AudioClassificationPipelineOutput(PipelineIO):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    results: list[dict[str, str | float]] = []

    _dict_keys = {frozenset({"label", "score"})}

    @field_validator("results", mode="after")
    @classmethod
    def _validate_results_value(
        cls, value: list[dict[str, str | float]]
    ) -> list[dict[str, str | float]]:
        if not isinstance(value, list):
            raise ValueError("Results must be a list")
        for item in value:
            if not isinstance(item, dict):
                raise ValueError("Results items must be dictionaries")
            if frozenset(item.keys()) not in cls._dict_keys:
                raise ValueError("Results items must have keys 'label' and 'score'")
            if not isinstance(item["label"], str):
                raise ValueError("Results item 'label' must be a string")
            if not isinstance(item["score"], float):
                raise ValueError("Results item 'score' must be a float")
        return value

