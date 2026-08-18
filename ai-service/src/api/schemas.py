from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# --- Documents ---


class IngestRequest(BaseModel):
    """Sent by Rust gateway alongside the uploaded file."""

    document_id: str = Field(min_length=1)


class IngestResponse(BaseModel):
    """Returned after synchronous ingestion."""

    document_id: str
    diagnosis_name: str | None
    mkb_code: str | None
    chunk_count: int
    full_text: str
    sections_json: str


class DeleteDocumentRequest(BaseModel):
    document_id: str = Field(min_length=1)


# --- Algorithms ---


class AlgorithmGenerate(BaseModel):
    full_text: str = Field(min_length=100)
    diagnosis_name: str = Field(min_length=1)
    mode: Literal["structured", "source", "physician"] = "physician"


class ExportPdfRequest(BaseModel):
    markdown: str = Field(min_length=100)


# --- Chat ---


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatAttachmentContext(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    document_id: str | None = Field(default=None)
    text: str = Field(min_length=1)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=10000)
    document_id: str = Field(min_length=1)
    diagnosis_name: str = Field(min_length=1)
    history: list[ChatMessage] = Field(default_factory=list)
    attachments: list[ChatAttachmentContext] = Field(default_factory=list)


class ChatAttachmentIngestResponse(BaseModel):
    document_id: str
    filename: str
    full_text: str
    chunk_count: int


# --- Services ---


class ServiceMatchRequest(BaseModel):
    step_text: str = Field(min_length=1, max_length=5000)
    step_title: str = ""


class ServiceItem(BaseModel):
    id: str
    name: str


class ServiceMatchResponse(BaseModel):
    services: list[ServiceItem]
    step_title: str


# --- Calculators ---


class BMIRequest(BaseModel):
    height_cm: float = Field(gt=0, le=300)
    weight_kg: float = Field(gt=0, le=500)


class CreatinineRequest(BaseModel):
    age: int = Field(gt=0, le=150)
    weight_kg: float = Field(gt=0, le=500)
    creatinine: float = Field(gt=0, le=2000)
    gender: str = Field(pattern=r"^(male|female)$")


class BSARequest(BaseModel):
    height_cm: float = Field(gt=0, le=300)
    weight_kg: float = Field(gt=0, le=500)


class DosageRequest(BaseModel):
    weight_kg: float = Field(gt=0, le=500)
    dose_per_kg: float = Field(gt=0, le=1000)
    frequency: int = Field(ge=1, le=24, default=1)


class ScoreRequest(BaseModel):
    age: int = Field(ge=40, le=65)
    gender: str = Field(pattern=r"^(male|female)$")
    smoking: bool
    cholesterol: float = Field(gt=0, le=20)
    systolic_bp: int = Field(ge=80, le=300)


class CalculatorResult(BaseModel):
    result: float
    unit: str
    interpretation: str
