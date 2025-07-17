from typing import Any

import numpy as np
from pydantic import field_validator, field_serializer

from icicle_playgrounds.pydantic.transformers.PipelineIO import PipelineIO

class TextToAudioPipelineInput(PipelineIO):
    text_inputs: str | list[str]
    forward_params: dict[str, Any] | None = None
    generate_kwargs: dict[str, Any] | None = None

class TextToAudioPipelineOutput(PipelineIO):
    results: dict[str, int | np.ndarray] | list[dict[str, int | np.ndarray]]

    # Private Attributes
    _ndims = 2
    _dtypes = {np.float32, np.float64}
    _dict_keys = {frozenset({"audio", "sampling_rate")}

    @field_validator("results", mode="after")
    @classmethod
    def _validate_results_value(
        cls, value: dict[str, int | np.ndarray] | list[dict[str, int | np.ndarray]]
    ) -> dict[str, int | np.ndarray] | list[dict[str, int | np.ndarray]]:
        if not isinstance(value, (dict, list)):
            raise ValueError("Results must be a dictionary or a list of dictionaries")
        results = [value] if isinstance(value, dict) else value
        for result in results:
            if frozenset(result.keys()) not in cls._dict_keys:
                raise ValueError("Results items must have keys 'audio' and 'sampling_rate'")
            if not isinstance(result["audio"], np.ndarray):
                raise ValueError("Results item 'audio' must be a numpy array")
            if not isinstance(result["sampling_rate"], int):
                raise ValueError("Results item 'sampling_rate' must be a integer")
            # Validate NumPy Array
            cls._validate_ndarray(result["audio"])

    @field_serializer("results")
    def _serialize_results(
            self,
            value: dict[str, int | np.ndarray] | list[dict[str, int | np.ndarray]],
            compress: bool = False
    ) -> dict[str, int | str] | list[dict[str, int | str]]:
        results = [value] if isinstance(value, dict) else value
        for result in results:
            result["audio"] = self._ndarray_to_base64(array=result["audio"], compress=compress)
        if isinstance(value, dict):
            results = results[0]
        return results
