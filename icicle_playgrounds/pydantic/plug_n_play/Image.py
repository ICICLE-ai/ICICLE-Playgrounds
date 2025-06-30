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
    """A versatile image handling class that supports multiple input formats and conversions.

    This class can handle images from various sources including URLs, file paths, base64 strings,
    NumPy arrays, PyTorch tensors, and PIL Images. It provides methods for converting between
    different image formats and serialization capabilities.

    Attributes:
        image (PIL.Image.Image): The internal PIL Image representation of the image data.

    Examples:
        >>> # From URL
        >>> img = Image(image="http://example.com/image.jpg")
        >>> 
        >>> # From file
        >>> img = Image(image="file:/path/to/image.png")
        >>> 
        >>> # From base64
        >>> img = Image(image="base64_encoded_string")
        >>> 
        >>> # From numpy array
        >>> img = Image(image=numpy_array)
        >>> 
        >>> # From PyTorch tensor
        >>> img = Image(image=torch_tensor)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: PILImage.Image

    def to_numpy(self) -> ndarray:
        """Converts the internal PIL Image to a NumPy array.

        Returns:
            ndarray: A NumPy array containing the image data with shape (H, W, C) for RGB
            images or (H, W) for grayscale images.
        """
        return np.asarray(self.image)

    def to_tensor(self) -> Tensor:
        """Convert the image to a PyTorch tensor.

        Returns:
            Tensor: The image data as a PyTorch tensor.
        """
        return PILToTensor()(self.image)

    @classmethod
    def __check_if_base64(cls, value: str) -> bool:
        """Check if a string is a valid base64 encoded value.

        Args:
            value (str): The string to check.

        Returns:
            bool: True if the string is valid base64, False otherwise.
        """
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
        """Create a PIL Image from a base64 encoded string.

        Args:
            value (str): The base64 encoded image string.

        Returns:
            PIL.Image.Image: The decoded image.

        Raises:
            ValueError: If the base64 string cannot be decoded to an image.
        """
        try:
            buffer = io.BytesIO(base64.b64decode(value))
            image = PILImage.open(buffer)
            return image
        except Exception:
            raise ValueError("Error decoding base64 string")

    @classmethod
    def __build_from_url(cls, url: str) -> PILImage.Image:
        """Creates a PIL Image by downloading from a URL.

        Args:
            url (str): The URL of the image to download. Must be a valid HTTP(S) URL
            pointing to an image file.

        Returns:
            PIL.Image.Image: The downloaded and decoded image.

        Raises:
            Exception: If the image cannot be downloaded or processed, including:
            - Network connectivity issues
            - Invalid URL
            - Non-image content
            - Server errors
        """
        try:
            response = httpx.get(url)
            response.raise_for_status()
            buffer = response.content
            return PILImage.open(io.BytesIO(buffer))
        except Exception as e:
            raise e

    @classmethod
    def __build_from_file(cls, path: str) -> PILImage.Image:
        """Create a PIL Image from a file path.

        Args:
            path (str): The path to the image file (can include 'file:/' prefix).

        Returns:
            PIL.Image.Image: The loaded image.

        Raises:
            Exception: If the file cannot be opened or processed.
        """
        try:
            path = path.replace("file:/", "")
            image = PILImage.open(path)
            return image
        except Exception as e:
            raise e

    @classmethod
    def __build_from_numpy(cls, value: ndarray) -> PILImage.Image:
        """Create a PIL Image from a NumPy array.

        Args:
            value (ndarray): The NumPy array containing image data.

        Returns:
            PIL.Image.Image: The converted image.

        Raises:
            ValueError: If the NumPy array cannot be converted to an image.
        """
        try:
            return ToPILImage()(value)
        except Exception:
            raise ValueError("Invalid NumPy array format")

    @classmethod
    def __build_from_tensor(cls, value: Tensor) -> PILImage.Image:
        """Create a PIL Image from a PyTorch tensor.

        Args:
            value (Tensor): The PyTorch tensor containing image data.

        Returns:
            PIL.Image.Image: The converted image.

        Raises:
            ValueError: If the tensor cannot be converted to an image.
        """
        try:
            return ToPILImage()(value)
        except Exception:
            raise ValueError("Invalid tensor format")

    @field_validator("image", mode="before")
    @classmethod
    def _validate_input_value(cls, value: Any) -> PILImage.Image:
        """Validates and converts input value to PIL Image.

        This validator handles multiple input formats including URLs, file paths,
        base64 strings, PIL Images, NumPy arrays, and PyTorch tensors.

        Args:
            value (Any): The input value to validate and convert. Can be one of:
            - str: URL starting with 'http', file path starting with 'file:/', or base64 string
            - PIL.Image.Image: Direct PIL Image instance
            - ndarray: NumPy array with shape (H, W, C) or (H, W)
            - Tensor: PyTorch tensor with shape (C, H, W)

        Returns:
            PIL.Image.Image: The validated and converted PIL Image instance.

        Raises:
            ValueError: If the input value cannot be converted to a valid image.
            Common cases include invalid string formats, malformed base64 data,
            or incompatible array shapes.
        """
        if isinstance(value, str):
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
    def _serialize_image(self, image: PILImage.Image) -> str:
        """Serialize the PIL Image to a base64 encoded string.

        Args:
            image (PIL.Image.Image): The image to serialize.

        Returns:
            str: The base64 encoded string representation of the image.
        """
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")