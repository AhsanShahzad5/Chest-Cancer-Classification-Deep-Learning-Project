"""Shapes of the API's JSON responses. FastAPI validates every reply against these
and shows them in /docs and /openapi.json."""

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    prediction: str = Field(examples=["Normal"], description="Predicted class")
    confidence: float = Field(
        ge=0,
        le=1,
        examples=[0.93],
        description="The model's score for the predicted class (0 to 1). "
        "It is not a guarantee that the prediction is right.",
    )
    model_version: str | None = Field(
        examples=["4"], description="Registry version of the model that answered"
    )


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])
    model_loaded: bool
    model_source: str | None = Field(examples=["models:/VGG16Model@champion"])
    model_version: str | None = Field(examples=["4"])
