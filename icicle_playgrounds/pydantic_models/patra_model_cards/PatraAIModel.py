from pydantic import BaseModel, Field

class PatraAIModel(BaseModel):
    backbone: str | None = Field(default=None, alias="Backbone")
    batch_size: int | None = Field(default=None, alias="Batch_Size")
    epochs: int | None = Field(default=None, alias="Epochs")
    input_shape: str | None = Field(default=None, alias="Input_Shape")
    learning_rate: float | None = Field(default=None, alias="Learning_Rate")
    optimizer: str | None = Field(default=None, alias="Optimizer")
    precision: float | None = Field(default=None, alias="Precision")
    recall: float | None = Field(default=None, alias="Recall")
    description: str
    foundational_model: str | None = None
    framework: str
    inference_labels: str
    license: str
    location: str
    model_id: str
    model_type: str
    name: str
    owner: str
    test_accuracy: float
    version: str

    class Config:
        populate_by_name = True