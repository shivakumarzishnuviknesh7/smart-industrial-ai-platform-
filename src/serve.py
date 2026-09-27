from contextlib import asynccontextmanager
from typing import List

import mlflow.pytorch
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

REGISTERED_MODEL_NAME = "industrial-baseline-cnn"

MODEL_ALIAS = "champion"

MODEL_URI = (
    f"models:/{REGISTERED_MODEL_NAME}"
    f"@{MODEL_ALIAS}"
)


# ---------------------------------------------------------
# FashionMNIST labels
# ---------------------------------------------------------

CLASS_NAMES = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}


# ---------------------------------------------------------
# Global model container
# ---------------------------------------------------------

model = None


# ---------------------------------------------------------
# API schemas
# ---------------------------------------------------------

class PredictionRequest(BaseModel):
    """
    Input must contain a 28 x 28 grayscale image.

    Pixel values should be between 0 and 255.
    """

    pixels: List[List[float]] = Field(
        ...,
        description=(
            "28x28 grayscale image "
            "with values from 0 to 255."
        ),
    )


class PredictionResponse(BaseModel):
    predicted_class: int

    predicted_label: str

    confidence: float

    model_name: str

    model_alias: str


# ---------------------------------------------------------
# Model loading
# ---------------------------------------------------------

def load_champion_model():
    """
    Load the model currently assigned
    to the MLflow 'champion' alias.
    """

    print(
        f"Loading model from: "
        f"{MODEL_URI}"
    )

    loaded_model = (
        mlflow.pytorch.load_model(
            MODEL_URI
        )
    )

    loaded_model.eval()

    print(
        "Champion model loaded successfully"
    )

    return loaded_model


# ---------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):

    global model

    model = load_champion_model()

    yield

    model = None


# ---------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------

app = FastAPI(
    title="Smart Industrial AI Platform",
    description=(
        "Inference API backed by the "
        "champion model stored in the "
        "MLflow Model Registry."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------

@app.get("/")
def root():

    return {
        "application":
            "Smart Industrial AI Platform",
        "status":
            "running",
    }


# ---------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------

@app.get("/health")
def health():

    if model is None:

        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    return {
        "status": "healthy",
        "model_loaded": True,
        "model":
            REGISTERED_MODEL_NAME,
        "alias":
            MODEL_ALIAS,
    }


# ---------------------------------------------------------
# Model information endpoint
# ---------------------------------------------------------

@app.get("/model")
def model_information():

    return {
        "registered_model":
            REGISTERED_MODEL_NAME,
        "alias":
            MODEL_ALIAS,
        "model_uri":
            MODEL_URI,
    }


# ---------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    request: PredictionRequest,
):

    if model is None:

        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    # -----------------------------------------------------
    # Validate image dimensions
    # -----------------------------------------------------

    if len(request.pixels) != 28:

        raise HTTPException(
            status_code=400,
            detail=(
                "Image must contain "
                "exactly 28 rows."
            ),
        )

    for row in request.pixels:

        if len(row) != 28:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Each image row must "
                    "contain exactly "
                    "28 pixels."
                ),
            )

    # -----------------------------------------------------
    # Convert input to tensor
    # -----------------------------------------------------

    image = torch.tensor(
        request.pixels,
        dtype=torch.float32,
    )

    # -----------------------------------------------------
    # Validate pixel range
    # -----------------------------------------------------

    if (
        torch.any(image < 0)
        or torch.any(image > 255)
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Pixel values must be "
                "between 0 and 255."
            ),
        )

    # -----------------------------------------------------
    # Match training preprocessing
    #
    # Training:
    # ToTensor() -> [0, 1]
    # Normalize(0.5, 0.5) -> [-1, 1]
    # -----------------------------------------------------

    image = image / 255.0

    image = (
        image - 0.5
    ) / 0.5

    # Shape:
    #
    # [28, 28]
    #      ↓
    # [1, 1, 28, 28]

    image = (
        image
        .unsqueeze(0)
        .unsqueeze(0)
    )

    # -----------------------------------------------------
    # Inference
    # -----------------------------------------------------

    model.eval()

    with torch.no_grad():

        logits = model(
            image
        )

        probabilities = (
            torch.softmax(
                logits,
                dim=1,
            )
        )

        confidence, prediction = (
            torch.max(
                probabilities,
                dim=1,
            )
        )

    predicted_class = (
        prediction.item()
    )

    predicted_label = (
        CLASS_NAMES[
            predicted_class
        ]
    )

    prediction_confidence = (
        confidence.item()
    )

    # -----------------------------------------------------
    # Response
    # -----------------------------------------------------

    return PredictionResponse(
        predicted_class=
            predicted_class,

        predicted_label=
            predicted_label,

        confidence=
            prediction_confidence,

        model_name=
            REGISTERED_MODEL_NAME,

        model_alias=
            MODEL_ALIAS,
    )