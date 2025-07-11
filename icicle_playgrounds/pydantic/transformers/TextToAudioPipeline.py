from typing import Any

import numpy as np
from pydantic import field_validator

from icicle_playgrounds.pydantic.transformers.PipelineIO import PipelineIO

class TextToAudioPipelineInput(PipelineIO):
    text_inputs: str | list[str]
    forward_params: dict[str, Any] | None = None
    generate_kwargs: dict[str, Any] | None = None

class TextToAudioPipelineOutput(PipelineIO):
    results: dict[str, int | np.ndarray] | list[dict[str, int | np.ndarray]]

    _ndims = 2
    _dtypes = {np.float32, np.float64}
    _dict_keys = {frozenset({"audio", "sampling_rate")}

    @field_validator("results", mode="after")
    @classmethod
    def _validate_results_value(
        cls, value: dict[str, np.ndarray | int] | list[dict[str, np.ndarray | int]]
    ) -> dict[str, np.array | int] | list[dict[str, np.array | int]]:
        if not isinstance(value, (dict, list)):
            raise ValueError("Results must be a dictionary or a list of dictionaries")
        if isinstance(value, dict):
            if not isinstance(value["audio"], np.ndarray):
                raise ValueError("Results 'audio' value must be a numpy array")