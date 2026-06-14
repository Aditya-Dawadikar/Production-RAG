from pydantic import BaseModel

class InferenceRequest(BaseModel):
    query: str


class InferenceResponse(BaseModel):
    query: str
    answer: str
    sources: list
    metadata: dict