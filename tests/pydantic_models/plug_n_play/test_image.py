import base64
import io

import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from PIL import Image as PILImage

from icicle_playgrounds.pydantic.plug_n_play.Image import Image


@pytest.fixture
def sample_rgb_array():
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_pil_image(sample_rgb_array):
    return PILImage.fromarray(sample_rgb_array)


@pytest.fixture
def sample_tensor():
    return torch.randint(0, 255, (3, 100, 100), dtype=torch.uint8)


@pytest.fixture
def sample_base64_image(sample_pil_image):
    buffer = io.BytesIO()
    sample_pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# Hypothesis strategies
@st.composite
def valid_rgb_arrays(draw):
    """Strategy to generate valid RGB numpy arrays"""
    height = draw(st.integers(min_value=1, max_value=100))
    width = draw(st.integers(min_value=1, max_value=100))
    return draw(
        arrays(
            dtype=np.uint8,
            shape=(height, width, 3),
            elements=st.integers(min_value=0, max_value=255),
        )
    )


@st.composite
def valid_grayscale_arrays(draw):
    """Strategy to generate valid grayscale numpy arrays"""
    height = draw(st.integers(min_value=1, max_value=100))
    width = draw(st.integers(min_value=1, max_value=100))
    return draw(
        arrays(
            dtype=np.uint8,
            shape=(height, width),
            elements=st.integers(min_value=0, max_value=255),
        )
    )


@st.composite
def valid_tensors(draw):
    """Strategy to generate valid PyTorch tensors"""
    height = draw(st.integers(min_value=1, max_value=100))
    width = draw(st.integers(min_value=1, max_value=100))
    channels = draw(st.integers(min_value=1, max_value=3))
    array = draw(
        arrays(
            dtype=np.uint8,
            shape=(channels, height, width),
            elements=st.integers(min_value=0, max_value=255),
        )
    )
    return torch.from_numpy(array)


class TestImage:
    @given(valid_rgb_arrays())
    def test_hypothesis_create_from_numpy_rgb(self, array):
        """Test that any valid RGB numpy array can be converted to an Image"""
        image = Image(image=array)
        assert isinstance(image.image, PILImage.Image)
        # Check that the conversion preserves the data
        np.testing.assert_array_almost_equal(np.array(image.image), array)

    @given(valid_grayscale_arrays())
    def test_hypothesis_create_from_numpy_grayscale(self, array):
        """Test that any valid grayscale numpy array can be converted to an Image"""
        image = Image(image=array)
        assert isinstance(image.image, PILImage.Image)
        # For grayscale, the output might be 2D or 3D with same values in each channel
        np_image = np.array(image.image)
        if len(np_image.shape) == 3:
            np.testing.assert_array_almost_equal(np_image[:, :, 0], array)
        else:
            np.testing.assert_array_almost_equal(np_image, array)

    @given(valid_tensors())
    def test_hypothesis_create_from_tensor(self, tensor):
        """Test that any valid tensor can be converted to an Image"""
        image = Image(image=tensor)
        assert isinstance(image.image, PILImage.Image)

    @given(st.text(alphabet=st.characters(blacklist_categories=("Cs",))))
    def test_hypothesis_invalid_string_input(self, invalid_str):
        """Test that invalid strings are properly rejected"""
        if not Image._Image__check_if_base64(
            invalid_str
        ) and not invalid_str.startswith(("http", "file")):
            with pytest.raises(ValueError):
                Image(image=invalid_str)

    def test_create_from_pil_image(self, sample_pil_image):
        image = Image(image=sample_pil_image)
        assert isinstance(image.image, PILImage.Image)

    def test_create_from_numpy(self, sample_rgb_array):
        image = Image(image=sample_rgb_array)
        assert isinstance(image.image, PILImage.Image)
        np.testing.assert_array_almost_equal(np.array(image.image), sample_rgb_array)

    def test_create_from_tensor(self, sample_tensor):
        image = Image(image=sample_tensor)
        assert isinstance(image.image, PILImage.Image)

    def test_create_from_base64(self, sample_base64_image):
        image = Image(image=sample_base64_image)
        assert isinstance(image.image, PILImage.Image)

    def test_to_numpy(self, sample_pil_image):
        image = Image(image=sample_pil_image)
        numpy_array = image.to_numpy()
        assert isinstance(numpy_array, np.ndarray)
        np.testing.assert_array_equal(numpy_array, np.array(sample_pil_image))

    def test_to_tensor(self, sample_pil_image):
        image = Image(image=sample_pil_image)
        tensor = image.to_tensor()
        assert isinstance(tensor, torch.Tensor)

    def test_create_from_url(self):
        image = Image(
            image="https://raw.githubusercontent.com/tapis-project/camera-traps/refs/heads/main/installer/example_images/baby-red-fox.jpg"
        )
        assert isinstance(image.image, PILImage.Image)

    def test_serialize_image(self, sample_pil_image):
        image = Image(image=sample_pil_image)
        serialized = image.model_dump()
        assert isinstance(serialized["image"], str)

        # Verify we can decode it back
        decoded = Image.model_validate(serialized)
        assert isinstance(decoded.image, PILImage.Image)

        serialized_json = image.model_dump_json()
        assert isinstance(serialized_json, str)

        # Verify we can decode it back
        decoded_json = Image.model_validate_json(serialized_json)
        assert isinstance(decoded_json.image, PILImage.Image)

    def test_invalid_string_input(self):
        with pytest.raises(ValueError, match="Invalid value string format"):
            Image(image="invalid_string")

    def test_invalid_numpy_array(self):
        invalid_array = np.array([1, 2, 3])  # 1D array
        with pytest.raises(ValueError, match="Invalid NumPy array format"):
            Image(image=invalid_array)

    def test_invalid_base64(self):
        with pytest.raises(ValueError, match="Invalid value string format"):
            Image(image="invalidbase64string")

    @given(
        arrays(
            dtype=np.uint8,
            shape=st.tuples(st.integers(1, 10), st.integers(1, 10), st.integers(5, 10)),
        )
    )
    def test_hypothesis_invalid_channel_count(self, array):
        """Test that arrays with too many channels are rejected"""
        with pytest.raises(ValueError):
            Image(image=array)

    @given(arrays(dtype=np.uint8, shape=st.tuples(st.integers(1, 10))))
    def test_hypothesis_invalid_dimensions(self, array):
        """Test that 1D arrays are rejected"""
        with pytest.raises(ValueError):
            Image(image=array)

    def test_invalid_url(self):
        """Test handling of invalid or non-existent URLs"""
        with pytest.raises(Exception):
            Image(image="http://nonexistent.example.com/image.jpg")

    def test_invalid_file_path(self):
        """Test handling of invalid file paths"""
        with pytest.raises(Exception):
            Image(image="file:/nonexistent/path/image.jpg")

    def test_corrupted_base64(self):
        """Test handling of corrupted base64 data"""
        # Valid base64 structure but corrupted image data
        corrupted_base64 = base64.b64encode(b"not an image").decode("utf-8")
        with pytest.raises(ValueError):
            Image(image=corrupted_base64)

    @given(st.binary(min_size=1))
    def test_hypothesis_invalid_image_data(self, invalid_data):
        """Test handling of invalid image data in various formats"""
        # Convert to base64 to test invalid image data in base64 format
        base64_data = base64.b64encode(invalid_data).decode("utf-8")
        if Image._Image__check_if_base64(base64_data):
            with pytest.raises(ValueError):
                Image(image=base64_data)

    @given(
        st.one_of(
            arrays(
                dtype=np.complex64,
                shape=st.tuples(
                    st.integers(1, 10), st.integers(1, 10), st.integers(1, 3)
                ),
            ),
            arrays(
                dtype=np.bool_,
                shape=st.tuples(
                    st.integers(1, 10), st.integers(1, 10), st.integers(1, 3)
                ),
            ),
        )
    )
    def test_hypothesis_invalid_dtype(self, array):
        """Test that arrays with invalid dtypes are rejected"""
        with pytest.raises(ValueError):
            Image(image=array)

    def test_image_size_preservation(self, sample_pil_image):
        """Test that image dimensions are preserved after conversion"""
        original_size = sample_pil_image.size
        image = Image(image=sample_pil_image)
        assert image.image.size == original_size

    def test_image_mode_preservation(self, sample_pil_image):
        """Test that image mode (RGB, RGBA, etc.) is preserved"""
        original_mode = sample_pil_image.mode
        image = Image(image=sample_pil_image)
        assert image.image.mode == original_mode

    @pytest.mark.parametrize("none_value", [None, ""])
    def test_none_input(self, none_value):
        """Test handling of None and None-like values"""
        image = Image(image=none_value)
        assert image.image is None

    def test_large_tensor_handling(self):
        """Test handling of large tensor inputs"""
        large_tensor = torch.randint(0, 255, (3, 1000, 1000), dtype=torch.uint8)
        image = Image(image=large_tensor)
        assert isinstance(image.image, PILImage.Image)
        assert image.image.size == (1000, 1000)

    def test_none_handling(self):
        """Test comprehensive handling of None values in different scenarios"""
        # Direct None input - should return None
        image = Image(image=None)
        assert image.image is None

        # Test None in numpy array
        array_with_none = np.array([[None, None], [None, None]])
        with pytest.raises(ValueError):
            Image(image=array_with_none)

        # Test None in tensor - skip this test as torch.tensor(None) is not valid
        # Test string representation of None
        with pytest.raises(ValueError):
            Image(image="None")

        # Test empty string - should return None
        image = Image(image="")
        assert image.image is None

        # Test string with only whitespace
        with pytest.raises(ValueError):
            Image(image="   ")
