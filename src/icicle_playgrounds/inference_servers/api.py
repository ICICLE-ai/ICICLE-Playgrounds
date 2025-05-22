from abc import ABC, abstractmethod
from litserve import LitAPI

class InferenceServerAPI(LitAPI, ABC):
    """
    Represents an inference server API that integrates batching and asynchronous processing capabilities.

    Provides a framework for interacting with an inference server, managing model-related operations,
    and handling configuration settings for efficient batch processing.

    :ivar _model: Represents the currently loaded model in the inference server.
    :type _model: object
    """
    def __init__(
        self,
        max_batch_size: int = 1,
        batch_timeout: float = 0.0,
        enable_async: bool = False,
    ):
        """
        Initializes an instance of the class with configurations for batch processing time limits, size constraints,
        and asynchronous enablement. The provided parameters define operational behavior for managing batch
        processing, enabling flexibility in different processing scenarios.

        :param max_batch_size: Specifies the maximum number of elements a single batch can contain.
        :type max_batch_size: int
        :param batch_timeout: Defines the timeout duration (in seconds) after which a batch will be processed,
            regardless of its size.
        :type batch_timeout: float
        :param enable_async: Indicates whether asynchronous operations are enabled for batch processing.
        :type enable_async: bool
        """
        super().__init__(max_batch_size, batch_timeout, enable_async)
        self._model = None

    @property
    def model(self):
        """
        This property retrieves the private attribute `_model`, representing the model
        associated with the current instance. The `_model` attribute is typically set
        during the instantiation of the object or updated through internal methods.
        Accessing this property allows read-only access to the model without directly
        exposing the private attribute.

        :return: The model associated with the current instance
        :rtype: Any
        """
        return self._model

    @model.setter
    def model(self, model):
        """
        Represents a setter method for the `model` property.

        This method allows updating the private `_model` attribute of the class,
        providing an interface for controlled property modification.

        :param model: Assigns a new value to the model property.
            Must conform to the expected data type or constraints.
        :type model: Any
        """
        self._model = model

    @model.deleter
    def model(self):
        """
        Deletes the internal `model` attribute.

        This method sets the `_model` attribute to `None`, effectively removing
        the reference to the stored model object and allowing it to be garbage
        collected if there are no other references to it.

        :raises AttributeError: If `_model` attribute is not defined.

        :return: None
        """
        self._model = None

    @abstractmethod
    def load_model(self, model_path) -> object:
        """
        Abstract method to load a model from the specified file path.

        This method should be implemented by any subclass to define the
        specific logic for loading a model. The type of the model loaded
        depends on the implementation in the subclass.

        :param model_path: Path to the model file that needs to be loaded.
                           This is used to locate the model on the file system.
        :type model_path: str
        :return: The loaded model object based on the implementation.
        :rtype: object
        """
        pass
