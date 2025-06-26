from pydantic import BaseModel

class PatraBiasAnalysis(BaseModel):
    external_id: str
    name: str