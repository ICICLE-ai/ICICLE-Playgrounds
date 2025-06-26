import io
import re
import base64
import httpx
import cv2
import numpy as np
from typing import Any
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, field_serializer

class Image(BaseModel):
    """
    Represents an image model for validating, building, and serializing image data.

    This class is designed to handle image data in various formats such as Base64-encoded strings,
    URLs, file paths, or NumPy arrays. It ensures that the input is converted into a valid
    NumPy array representing the image. The model also provides functionality to serialize images
    into Base64 format for easier storage or transmission.

    :ivar image: The image data represented as a NumPy array.
    :type image: np.ndarray
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    image: np.ndarray
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
    def __check_if_valid_numpy_image_array(cls, array: np.ndarray) -> bool:
        return (len(array.shape) == 3 and array.shape[-1] == 3) or len(array.shape) == 2

    @classmethod
    def __build_from_base64(cls, value: str) -> np.ndarray:
        try:
            image_buff = base64.b64decode(value)
            image_ndarray = np.frombuffer(image_buff, dtype=np.uint8)
            image = cv2.imdecode(image_ndarray, cv2.IMREAD_UNCHANGED)

            if cls.__check_if_valid_numpy_image_array(image):
                return image
            raise ValueError("Invalid value format")
        except Exception:
            raise ValueError("Invalid value format")


    @classmethod
    def __build_from_url(cls, url: str) -> np.ndarray:
        try:
            image_buff = httpx.get(url).content
            image = cv2.imdecode(np.frombuffer(image_buff, np.uint8), cv2.IMREAD_UNCHANGED)

            if cls.__check_if_valid_numpy_image_array(image):
                return image
            raise ValueError("Invalid value format")
        except Exception:
            raise ValueError("Invalid value format")

    @classmethod
    def __build_from_file(cls, path: str) -> np.ndarray:
        try:
            # Step 1: Load value from file path
            image = cv2.imread(path)
            if image is None:
                raise ValueError("Could not load value from file")

            if cls.__check_if_valid_numpy_image_array(image):
                return image
            raise ValueError("Invalid value format")
        except Exception:
            raise ValueError("Invalid value format")


    @field_validator("image", mode="before")
    @classmethod
    def validate_input_value(cls, value: Any) -> np.ndarray:
        if isinstance(value, str):
            # If it's not a NumPy array, then build one using the provided string.
            if cls.__check_if_base64(value):
                return cls.__build_from_base64(value)
            elif value.startswith("http"):
                return cls.__build_from_url(value)
            elif value.startswith("file"):
                return cls.__build_from_file(value)
            else:
                raise ValueError("Invalid value format")
        elif isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.uint8):
            if cls.__check_if_valid_numpy_image_array(value):
                return value
            raise ValueError("Invalid value format")
        else:
            raise ValueError("Invalid value format")

    @field_serializer("image")
    def serialize_image(self, image: np.ndarray) -> str:
        buffer = io.BytesIO()
        np.save(buffer, image)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")