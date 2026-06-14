# src/main.py

import logging

from fastapi import FastAPI, HTTPException
from src.data_models.Inference import InferenceRequest, InferenceResponse
from fastapi.middleware.cors import CORSMiddleware

from src.rag import rag_client

logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


@app.post(
    "/inference",
    response_model=InferenceResponse,
)
def inference(request: InferenceRequest):
    try:
        return rag_client.answer(
            query=request.query,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except Exception:
        logger.exception("Inference failed")
        raise HTTPException(
            status_code=500,
            detail="Internal error",
        )
