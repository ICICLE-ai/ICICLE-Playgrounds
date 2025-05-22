import cv2
import numpy as np

from pydantic import BaseModel

class Image(BaseModel):
    url: str | None = None
    file_path: str | None = None
    color_space: str | None = None