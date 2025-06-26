from pydantic import BaseModel

class PatraXAIAnalysis(BaseModel):
    external_id: str
    name: str