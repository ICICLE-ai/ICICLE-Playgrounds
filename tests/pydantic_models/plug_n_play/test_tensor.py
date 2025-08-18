import pytest
import numpy as np
import torch

from icicle_playgrounds.pydantic.plug_n_play.Tensor import Tensor



class TestTensor:
    test_int_array: list = [[1,2,3],[4,5,6],[7,8,9]]
    test_float_array: list = [[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]]

    test_int_np_dtype: np.dtype = np.int32
    test_float_np_dtype: np.dtype = np.float32

    test_int_torch_dtype: torch.dtype = torch.int32
    test_float_torch_dtype: torch.dtype = torch.float32

    """Test cases for the Tensor class."""
    def test_init_from_numpy_array_int(self):
        array = np.array(self.test_int_array, dtype=self.test_int_np_dtype)
        tensor = Tensor(data=array)

        assert isinstance(tensor.data, np.ndarray)
        assert tensor.data.dtype == self.test_int_np_dtype
        assert tensor.data.shape == (3,3)
        assert tensor.data.tolist() == self.test_int_array

        print(tensor.model_dump())
        print(tensor.model_dump_json())
        print(tensor.model_validate(tensor.model_dump()))
        print(tensor.model_validate_json(tensor.model_dump_json()))

        print(tensor.model_dump(context={"to_base64": True}))
        print(tensor.model_dump_json(context={"to_base64": True}))
        print(tensor.model_validate(tensor.model_dump(context={"to_base64": True})))
        print(
            tensor.model_validate_json(
                tensor.model_dump_json(context={"to_base64": True})
            )
        )

        print(tensor.model_dump(context={"to_base64": True, "compress": True}))
        print(tensor.model_dump_json(context={"to_base64": True, "compress": True}))
        print(
            tensor.model_validate(
                tensor.model_dump(context={"to_base64": True, "compress": True})
            )
        )
        print(
            tensor.model_validate_json(
                tensor.model_dump_json(context={"to_base64": True, "compress": True})
            )
        )

    def test_init_from_numpy_array_float(self):
        array = np.array(self.test_float_array, dtype=self.test_float_np_dtype)
        tensor = Tensor(data=array)
        assert tensor.data.dtype == self.test_float_np_dtype
        assert tensor.data.shape == (3,3)
        assert tensor.data.tolist() == self.test_float_array

        print(tensor.model_dump())
        print(tensor.model_dump_json())
        print(tensor.model_validate(tensor.model_dump()))
        print(tensor.model_validate_json(tensor.model_dump_json()))

        print(tensor.model_dump(context={"to_base64": True}))
        print(tensor.model_dump_json(context={"to_base64": True}))
        print(tensor.model_validate(tensor.model_dump(context={"to_base64": True})))
        print(tensor.model_validate_json(tensor.model_dump_json(context={"to_base64": True})))

        print(tensor.model_dump(context={"to_base64": True, "compress": True}))
        print(tensor.model_dump_json(context={"to_base64": True, "compress": True}))
        print(tensor.model_validate(tensor.model_dump(context={"to_base64": True, "compress": True})))
        print(tensor.model_validate_json(tensor.model_dump_json(context={"to_base64": True, "compress": True})))

    def test_init_from_torch_tensor_int(self):
        tensor = Tensor(data=torch.tensor(self.test_int_array, dtype=self.test_int_torch_dtype))
        assert isinstance(tensor.data, torch.Tensor)
        assert tensor.data.dtype == self.test_int_torch_dtype
        assert tensor.data.shape == (3,3)
        assert tensor.data.tolist() == self.test_int_array

        print(tensor.model_dump())
        print(tensor.model_dump_json())
        print(tensor.model_validate(tensor.model_dump()))
        print(tensor.model_validate_json(tensor.model_dump_json()))

        print(tensor.model_dump(context={"to_base64": True}))
        print(tensor.model_dump_json(context={"to_base64": True}))
        print(tensor.model_validate(tensor.model_dump(context={"to_base64": True})))
        print(
            tensor.model_validate_json(
                tensor.model_dump_json(context={"to_base64": True})
            )
        )

        print(tensor.model_dump(context={"to_base64": True, "compress": True}))
        print(tensor.model_dump_json(context={"to_base64": True, "compress": True}))
        print(
            tensor.model_validate(
                tensor.model_dump(context={"to_base64": True, "compress": True})
            )
        )
        print(
            tensor.model_validate_json(
                tensor.model_dump_json(context={"to_base64": True, "compress": True})
            )
        )

    def test_init_from_torch_tensor_float(self):
        tensor = Tensor(data=torch.tensor(self.test_float_array, dtype=self.test_float_torch_dtype))
        assert isinstance(tensor.data, torch.Tensor)
        assert tensor.data.dtype == self.test_float_torch_dtype
        assert tensor.data.shape == (3,3)
        assert tensor.data.tolist() == self.test_float_array

        print(tensor.model_dump())
        print(tensor.model_dump_json())
        print(tensor.model_validate(tensor.model_dump()))
        print(tensor.model_validate_json(tensor.model_dump_json()))
        print(tensor)
        print(tensor.model_dump(context={"to_base64": True}))
        print(tensor.model_dump_json(context={"to_base64": True}))
        print(tensor.model_validate(tensor.model_dump(context={"to_base64": True})))
        print(
            tensor.model_validate_json(
                tensor.model_dump_json(context={"to_base64": True})
            )
        )

        print(tensor.model_dump(context={"to_base64": True, "compress": True}))
        print(tensor.model_dump_json(context={"to_base64": True, "compress": True}))
        print(
            tensor.model_validate(
                tensor.model_dump(context={"to_base64": True, "compress": True})
            )
        )
        print(
            tensor.model_validate_json(
                tensor.model_dump_json(context={"to_base64": True, "compress": True})
            )
        )


if __name__ == "__main__":
    pytest.main([__file__])