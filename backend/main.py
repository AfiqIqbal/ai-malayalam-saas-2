import os
import shutil
import uuid
from pathlib import Path
from typing import Dict, List, Optional
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import RAG pipeline
from ai.rag_pipeline import MalayalamRAGPipeline

# Initialize FastAPI app
app = FastAPI(
    title="Malayalam AI Customer Service API",
    description="REST API for Malayalam language AI customer service",
    version="0.1.0"
)

# Configuration
BASE_DIR = Path(__file__).parent.parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
VECTOR_STORE_DIR = BASE_DIR / "data" / "vector_store"
MODEL_CACHE_DIR = BASE_DIR / "data" / "models"

# Create necessary directories
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Initialize RAG pipeline
rag_pipeline = MalayalamRAGPipeline()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class QueryRequest(BaseModel):
    text: str = Field(..., description="The query text in Malayalam or English")
    conversation_id: Optional[str] = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique conversation ID for tracking"
    )

class QueryResponse(BaseModel):
    response: str = Field(..., description="The AI's response in Malayalam")
    conversation_id: str = Field(..., description="The conversation ID")
    sources: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Sources used to generate the response"
    )

class UploadResponse(BaseModel):
    filename: str
    status: str
    vector_store_updated: bool

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Welcome to Malayalam AI Customer Service API",
        "status": "operational",
        "version": "0.1.0"
    }

@app.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def process_query(query: QueryRequest):
    """
    Process a natural language query and return an AI-generated response in Malayalam.
    """
    try:
        # Process the query using RAG pipeline
        response_text = rag_pipeline.process_query(query.text)
        
        return QueryResponse(
            response=response_text,
            conversation_id=query.conversation_id
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

@app.post("/upload/", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_knowledge_base(
    file: UploadFile = File(..., description="Document file to process (PDF or TXT)"),
    vector_store_name: str = Form("default", description="Name for the vector store")
):
    """
    Upload a document to the knowledge base.
    The document will be processed and added to the vector store.
    """
    try:
        # Validate file type
        if not (file.filename.endswith('.pdf') or file.filename.endswith('.txt')):
            raise HTTPException(
                status_code=400,
                detail="Only PDF and TXT files are supported"
            )
        
        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the document
        documents = rag_pipeline.load_documents(str(file_path))
        vector_store_path = VECTOR_STORE_DIR / vector_store_name
        rag_pipeline.create_vector_store(documents, str(vector_store_path))
        
        return UploadResponse(
            filename=file.filename,
            status="processed",
            vector_store_updated=True
        )
        
    except Exception as e:
        # Clean up if there was an error
        if 'file_path' in locals() and file_path.exists():
            file_path.unlink()
        
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "vector_store_initialized": rag_pipeline.vector_store is not None,
        "model_loaded": True  # TODO: Add actual model loading check
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1
    )
