from pydantic import BaseModel
from .PatraAIModel import PatraAIModel
from .PatraBiasAnalysis import PatraBiasAnalysis
from .PatraXAIAnalysis import PatraXAIAnalysis

class PatraModelCard(BaseModel):
    ai_model: PatraAIModel
    author: str
    bias_analysis: PatraBiasAnalysis | None = None
    categories: str
    external_id: str
    full_description: str
    input_data: str
    input_type: str
    keywords: str
    name: str
    output_data: str
    short_description: str
    version: str
    xai_analysis: PatraXAIAnalysis | None = None