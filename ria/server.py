import uuid
import io
import os
import logging
import json

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import (
    BaseModel,
    confloat
)
from enum import Enum
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.responses import PlainTextResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.staticfiles import StaticFiles
from base64 import b64encode

from .ria_model import RIAModel
from .validation_model import ValidationModel
from .visualization import OutputVisualiztion

app = FastAPI()
log = logging.getLogger("app")

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"version": "0.1","name":"Respiratory Illness Assessment REST API"}


@app.get("/health")
async def system_health():
    return {"status": "ok"}


@app.exception_handler(StarletteHTTPException)
async def starlette_http_exception_handler(request, exc):
    log.error(f"ERROR:{exc}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": f"{exc.detail}"},
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    log.error(f"ERROR:{exc}")
    return JSONResponse(
        status_code=400,
        content={"message": f"{str(exc)}"},
    )


################################ Respiratory Illness Assessment Service ##########################################

class Audio(BaseModel):
    content: str = None
    uri: str = None

class RIARequest(BaseModel):
    audio: Audio

class RIAStatus(str, Enum):
    unknown = 'UNKNOWN'
    normal = 'NORMAL'
    covid = 'COVID'

class RIAResult(BaseModel):
    prediction: RIAStatus
    confidence: confloat(ge=0.0, le=1.0)

class ValidationStatus(str, Enum):
    unknown = 'UNKNOWN'
    correct = 'VALID'
    incorrect = 'INVALID'

class RIAResponse(BaseModel):
    status: ValidationStatus
    result: RIAResult = None

class VisualizationResponse(BaseModel):
    html_div: str

ria_model = RIAModel()
validation_model = ValidationModel()

@app.post("/api/health/v1/ria", response_model=RIAResponse)
# async def respiratory_illness_assessment(
#     request: RIARequest
#     ):
async def respiratory_illness_assessment(
    audio: UploadFile = File(...)
    ):
    b64_input = b64encode(audio.file.read()).decode('utf-8')

    validation_result = validation_model.predict(b64_input)

    if (validation_result["status"] == 'INVALID'):
        response = RIAResponse(
            status = ValidationStatus(validation_result["status"]),
            result = None
            )
    else:

        result = ria_model.predict(b64_input)

        response = RIAResponse(
            status=ValidationStatus(validation_result["status"]),
            result=RIAResult(prediction= RIAStatus(result["prediction"]),
                             confidence= result["confidence"])
            )

    return response

visualization_object = OutputVisualiztion()

@app.post("/api/health/v1/visualization", response_model=VisualizationResponse)
# async def respiratory_illness_assessment(
#     request: RIARequest
#     ):
async def audioVisualization(
    audio: UploadFile = File(...)
    ):
    b64_input = b64encode(audio.file.read()).decode('utf-8')

    output_html_div_string = visualization_object.plot(b64_input)

    response = VisualizationResponse(
        html_div = output_html_div_string
        )
    return response

app.mount("/", StaticFiles(directory="ria/static"), name="static")
