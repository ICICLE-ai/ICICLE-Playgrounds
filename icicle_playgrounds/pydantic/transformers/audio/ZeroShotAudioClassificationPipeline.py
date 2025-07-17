import numpy as np
from pydantic import field_validator

from icicle_playgrounds.pydantic.transformers.PipelineIO import PipelineIO


class ZeroShotAudioClassificationPipelineInput(PipelineIO):
    audios: str | list[str] | np.ndarray | list[np.ndarray]
    candidate_labels: str | list[str]
    hypothesis_template: str | None = None

    # Private Attributes
    _ndims = 1
    _dtypes = {np.float32, np.float64}

    @field_validator("audios", mode="before")
    @classmethod
    def _prevalidate_audios_value(
        cls, value: str | list[str] | np.ndarray | list[np.ndarray]
    ) -> str | list[str] | np.ndarray | list[np.ndarray]:
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
            if all(isinstance(val, str) for val in value):
                if all(val.startswith("file:/") for val in value):
                    value = [val.split("file:/")[1] for val in value]
                elif all(val.startswith("base64") for val in value):
                    value = [cls._base64_to_ndarray(encoded_str=val) for val in value]
            else:
                value = np.array(value)
        return value

    @field_validator("audios", mode="after")
    @classmethod
    def _validate_audios_value(
        cls, value: str | list[str] | np.ndarray | list[np.ndarray]
    ) -> str | list[str] | np.ndarray | list[np.ndarray]:
        if value is None:
            raise ValueError("Inputs cannot be None")
        if isinstance(value, (str, str)):
            return value
        elif isinstance(value, np.ndarray):
            # Need to validate numpy array dimensions and dtype
            cls._validate_ndarray_input(value)
            return value
        elif isinstance(value, list):
            if all([isinstance(val, str) for val in value]):
                return value
            elif all(isinstance(val, np.ndarray) for val in value):
                for val in value:
                    cls._validate_ndarray_input(val)
                return value
            else:
                raise ValueError("Invalid inputs format")
        else:
            raise ValueError("Invalid inputs format")

class ZeroShotAudioClassificationPipelineOutput(PipelineIO):
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

