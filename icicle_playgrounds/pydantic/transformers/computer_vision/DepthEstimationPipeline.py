from PIL.Image import Image
from icicle_playgrounds.pydantic.transformers.PipelineIO import PipelineIO

class DepthEstimationPipelineInput(PipelineIO):
    inputs: str | list[str] | Image | list[Image]
    parameters: dict[str, Any]
    timeout: float