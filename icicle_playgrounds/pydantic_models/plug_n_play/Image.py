import io
import re
import base64
import httpx
from PIL import Image as PILImage
import numpy as np
from numpy import ndarray
from torch import Tensor
from torchvision.transforms import PILToTensor, ToPILImage
from typing import Any
from pydantic import BaseModel, ConfigDict, field_validator, field_serializer

class Image(BaseModel):
    """
    Represents an image model for validating, building, and serializing image data.

    This class is designed to handle image data in various formats such as Base64-encoded strings,
    URLs, file paths, or NumPy arrays. It ensures that the input is converted into a valid
    NumPy image representing the image. The model also provides functionality to serialize images
    into Base64 format for easier storage or transmission.

    :ivar image: The image data represented as a NumPy image.
    :type image: PILImage.Image
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: PILImage.Image

    def to_numpy(self) -> ndarray:
        return np.asarray(self.image)

    def to_tensor(self) -> Tensor:
        return PILToTensor()(self.image)

    @classmethod
    def __check_if_base64(cls, value: str) -> bool:
        pattern = r"^[A-Za-z0-9+/]*[=]{0,2}$"
        if not re.match(pattern, value):
            return False
        try:
            decoded = base64.b64decode(value)
            encoded = base64.b64encode(decoded).decode()
            return value.rstrip("=") == encoded.rstrip("=")
        except Exception:
            return False

    @classmethod
    def __build_from_base64(cls, value: str) -> PILImage.Image:
        try:
            buffer = io.BytesIO(base64.b64decode(value))
            image = PILImage.open(buffer)
            return image
        except Exception:
            raise ValueError("Error decoding base64 string")


    @classmethod
    def __build_from_url(cls, url: str) -> PILImage.Image:
        try:
            response = httpx.get(url)
            response.raise_for_status()

            buffer = response.content

            return PILImage.open(io.BytesIO(buffer))
        except Exception as e:
            raise e

    @classmethod
    def __build_from_file(cls, path: str) -> PILImage.Image:
        try:
            # Step 1: Strip 'file:/' from string
            path = path.replace("file:/", "")

            # Step 2: Open image from file path
            image = PILImage.open(path)
            return image
        except Exception as e:
            raise e

    @classmethod
    def __build_from_numpy(cls, value: ndarray) -> PILImage.Image:
        try:
            # return PILImage.fromarray(value)
            return ToPILImage()(value)
        except Exception:
            raise ValueError("Invalid NumPy array format")

    @classmethod
    def __build_from_tensor(cls, value: Tensor) -> PILImage.Image:
        try:
            return ToPILImage()(value)
        except Exception:
            raise ValueError("Invalid tensor format")

    @field_validator("image", mode="before")
    @classmethod
    def validate_input_value(cls, value: Any) -> PILImage.Image:
        if isinstance(value, str):
            # If it's not a NumPy image, then build one using the provided string.
            if value.startswith("http"):
                return cls.__build_from_url(value)
            elif value.startswith("file"):
                return cls.__build_from_file(value)
            elif cls.__check_if_base64(value):
                return cls.__build_from_base64(value)
            else:
                raise ValueError("Invalid value string format")
        elif isinstance(value, PILImage.Image):
            return value
        elif isinstance(value, ndarray):
            return cls.__build_from_numpy(value)
        elif isinstance(value, Tensor):
            return cls.__build_from_tensor(value)
        else:
            raise ValueError("Invalid value format")

    @field_serializer("image")
    def serialize_image(self, image: PILImage.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")