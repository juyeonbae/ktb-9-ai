from pydantic import BaseModel

class ImageModel(BaseModel):
    filename: str
    status: str
