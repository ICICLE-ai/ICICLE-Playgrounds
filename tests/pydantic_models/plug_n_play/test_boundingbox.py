import pytest
import numpy as np
import torch
from icicle_playgrounds.pydantic.plug_n_play.BoundingBox import BoundingBox, BoundingBoxFormat


class TestBoundingBox:
    @pytest.fixture
    def sample_box(self):
        return np.array([10, 20, 30, 40], dtype=np.float32)

    def test_create_bounding_box(self, sample_box):
        bbox = BoundingBox(box=sample_box)
        assert isinstance(bbox.box, np.ndarray)
        np.testing.assert_array_equal(bbox.box, sample_box)
        assert bbox.format == BoundingBoxFormat.XYXY

    def test_create_bounding_box_with_format(self, sample_box):
        bbox = BoundingBox(box=sample_box, format=BoundingBoxFormat.XYWH)
        assert bbox.format == BoundingBoxFormat.XYWH

    def test_to_tensor(self, sample_box):
        bbox = BoundingBox(box=sample_box)
        tensor = bbox.to_tensor()
        assert isinstance(tensor, torch.Tensor)
        np.testing.assert_array_equal(tensor.numpy(), sample_box)

    def test_ndarray_to_base64(self, sample_box):
        encoded = BoundingBox._ndarray_to_base64(sample_box)
        assert isinstance(encoded, str)
        assert encoded.startswith("base64:")
        
        # Test decoding
        decoded = BoundingBox._base64_to_ndarray(encoded)
        np.testing.assert_array_equal(decoded, sample_box)

    def test_ndarray_to_base64_compressed(self, sample_box):
        encoded = BoundingBox._ndarray_to_base64(sample_box, compress=True)
        assert isinstance(encoded, str)
        assert encoded.startswith("base64-compressed:")
        
        # Test decoding
        decoded = BoundingBox._base64_to_ndarray(encoded)
        np.testing.assert_array_equal(decoded, sample_box)

    def test_validate_input_numpy(self, sample_box):
        result = BoundingBox._validate_input_value(sample_box)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, sample_box)

    def test_validate_input_tensor(self, sample_box):
        tensor = torch.from_numpy(sample_box)
        result = BoundingBox._validate_input_value(tensor)
        assert isinstance(result, torch.Tensor)
        np.testing.assert_array_equal(result.numpy(), sample_box)

    def test_validate_input_list(self):
        input_list = [10, 20, 30, 40]
        result = BoundingBox._validate_input_value(input_list)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array(input_list))

    def test_validate_input_invalid_type(self):
        with pytest.raises(TypeError):
            BoundingBox._validate_input_value(42)

    def test_validate_input_invalid_base64(self):
        with pytest.raises(ValueError):
            BoundingBox._validate_input_value("invalid:base64string")

    def test_serialize_box_to_json(self, sample_box):
        bbox = BoundingBox(box=sample_box)
        serialized = bbox.model_dump_json()
        assert isinstance(serialized, str)
        
        # Verify we can decode it back
        decoded = BoundingBox.model_validate_json(serialized)
        assert isinstance(decoded.box, np.ndarray)
        np.testing.assert_array_equal(decoded.box, sample_box)
        assert decoded.format == BoundingBoxFormat.XYXY

    def test_serialize_box_to_dict(self, sample_box):
        bbox = BoundingBox(box=sample_box)
        serialized = bbox.model_dump()
        assert isinstance(serialized["box"], str)
        assert isinstance(serialized["format"], str)
        assert serialized["format"] == BoundingBoxFormat.XYXY
        
        # Verify we can decode it back
        decoded = BoundingBox.model_validate(serialized)
        assert isinstance(decoded.box, np.ndarray)
        np.testing.assert_array_equal(decoded.box, sample_box)

    def test_serialize_none_box(self):
        bbox = BoundingBox(box=None)
        serialized = bbox.model_dump()
        assert serialized["box"] == ""
        
        # Verify we can decode it back
        decoded = BoundingBox.model_validate(serialized)
        assert decoded.box is None

    def test_serialize_with_different_format(self, sample_box):
        bbox = BoundingBox(box=sample_box, format=BoundingBoxFormat.XYWH)
        serialized = bbox.model_dump_json()
        
        # Verify we can decode it back
        decoded = BoundingBox.model_validate_json(serialized)
        assert decoded.format == BoundingBoxFormat.XYWH
        np.testing.assert_array_equal(decoded.box, sample_box)

    def test_serialize_complex_array(self):
        # Test with multi-dimensional array
        complex_box = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
        bbox = BoundingBox(box=complex_box)
        serialized = bbox.model_dump()
        
        # Verify we can decode it back
        decoded = BoundingBox.model_validate(serialized)
        np.testing.assert_array_equal(decoded.box, complex_box)

    def test_serialize_different_dtypes(self):
        # Test with different numpy dtypes
        dtypes_to_test = [np.int32, np.int64, np.float32, np.float64]
        for dtype in dtypes_to_test:
            box = np.array([1, 2, 3, 4], dtype=dtype)
            bbox = BoundingBox(box=box)
            serialized = bbox.model_dump()
            
            # Verify we can decode it back
            decoded = BoundingBox.model_validate(serialized)
            np.testing.assert_array_equal(decoded.box, box)
            assert decoded.box.dtype == box.dtype

    def test_roundtrip_serialization(self, sample_box):
        # Test complete roundtrip with multiple serialization/deserialization cycles
        original = BoundingBox(box=sample_box)
        
        # First round
        serialized1 = original.model_dump_json()
        decoded1 = BoundingBox.model_validate_json(serialized1)
        
        # Second round
        serialized2 = decoded1.model_dump_json()
        decoded2 = BoundingBox.model_validate_json(serialized2)
        
        # Verify data integrity after multiple rounds
        np.testing.assert_array_equal(original.box, decoded2.box)
        assert original.format == decoded2.format