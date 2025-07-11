from typing import Any

from numpy import ndarray, float32, float64
from pydantic import BaseModel, ConfigDict, field_validator
from icicle_playgrounds.utils.base64 import numpy_to_base64
from icicle_playgrounds.pydantic.transformers.audio.utils import _validate_ndarray_input


class ZeroShotAudioClassificationPipelineInput(BaseModel):
    audios: str | list[str] | ndarray | list[ndarray]
    candidate_labels: str | list[str]
    hypothesis_template: str | None = None

    @field_validator(field="audios", mode="before")
    @classmethod
    def _prevalidate_audios_value(
        cls, value: Any
    ) -> str | list[str] | ndarray | list[ndarray]:
        if value is None:
            raise ValueError("Audios cannot be None")
        if isinstance(value, str):
            if value.startswith("file:/"):
                return value.split("file:/")[1]
            return value
        elif isinstance(value, ndarray):
            _validate_ndarray_input(value)
        elif isinstance(value, list):
            # Check if all elements are strings
            if isinstance(value[0], ndarray):
                if not all(isinstance(x, ndarray) for x in value):
                    raise ValueError("All elements in the list must be ndarrays")
                else:
                    for x in value:
                        _validate_ndarray_input(x)
            elif isinstance(value[0], str):
                if not all(isinstance(x, str) for x in value):
                    raise ValueError("All elements in the list must be strings")
        else:
            raise ValueError(
                "Audios must be a string, a list of strings, a numpy array, or list of numpy arrays"
            )
        return value


class ZeroShotAudioClassificationPipelineOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    results: dict[str, ndarray | int] | list[dict[str, ndarray | int]] | None = None

    @classmethod
    def _validate_results_dictionary(
        cls, results_dict: dict[str, ndarray | int]
    ) -> None:
        if results_dict.keys() != {"sampling_rate", "audio"}:
            raise ValueError(
                "Results results_dict must have keys 'sampling_rate' and 'audio'"
            )
        if not isinstance(results_dict["sampling_rate"], int):
            raise ValueError("Sampling rate must be an integer")
        if not isinstance(results_dict["audio"], ndarray):
            raise ValueError("Raw audio must be a numpy array")
        if results_dict["audio"].ndim != 2:
            raise ValueError("Raw audio must be a 2D array")
        if (
            results_dict["audio"].dtype != float32
            or results_dict["audio"].dtype != float64
        ):
            raise ValueError(
                "Raw audio must be a float array of 32 or 64 bit precision"
            )

    @field_validator("results", mode="after")
    @classmethod
    def _validate_result_value(
        cls, value: Any
    ) -> dict[str, ndarray | int] | list[dict[str, ndarray | int]] | None:
        if value is None:
            return None
        if isinstance(value, dict):
            cls._validate_results_dictionary(value)
        elif isinstance(value, list):
            for item in value:
                if not isinstance(item, dict):
                    raise ValueError("Result items must be dictionaries")
                cls._validate_results_dictionary(item)
        else:
            raise ValueError("Result must be a results_dict or a list of dictionaries")
        return value

    @field_serializer("results")
    def _serialize_results(
        self,
        results: dict[str, ndarray | int] | list[dist[str, ndarray | int]] | None,
        compress=False,
    ) -> str | dict[str, str | int] | list[dict[str, str | int]]:
        model_dump = None
        if results is None:
            model_dump = None
        elif isinstance(results, dict):
            model_dump = {
                "sampling_rate": results["sampling_rate"],
                "audio": numpy_to_base64(results["audio"], compress=compress),
            }
        elif isinstance(results, list):
            model_dump = []
            for item in results:
                model_dump.append(
                    {
                        "sampling_rate": item["sampling_rate"],
                        "audio": numpy_to_base64(item["audio"], compress=compress),
                    }
                )
        else:
            raise ValueError("Result must be a dictionary or a list of dictionaries")
        return model_dump
