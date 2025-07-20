"""FastAPI Application for Text-to-JSON Conversion System"""

import os
import json
import asyncio
from typing import Dict, Any, List, Optional
from pathlib import Path
import tempfile
import io

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import our core modules
from ..core.llm_manager import LLMManager, LLMConfig, ConversionResult
from ..core.schema_analyzer import SchemaAnalyzer, SchemaMetrics, ProcessingStrategy
from ..core.text_processor import TextProcessor, ProcessedText


# Pydantic models for API
class ConversionRequest(BaseModel):
    input_text: str = Field(..., description="Input text to convert")
    schema: Dict[str, Any] = Field(..., description="JSON schema to follow")
    input_format: Optional[str] = Field(None, description="Input format hint")


class SchemaAnalysisRequest(BaseModel):
    schema: Dict[str, Any] = Field(..., description="JSON schema to analyze")


class ConversionResponse(BaseModel):
    success: bool
    json_output: Dict[str, Any]
    confidence_score: float
    processing_time: float
    api_calls_made: int
    validation_errors: List[str]
    metadata: Dict[str, Any]


class SchemaAnalysisResponse(BaseModel):
    metrics: Dict[str, Any]
    strategy: Dict[str, Any]
    estimated_token_usage: Dict[str, int]


class HealthResponse(BaseModel):
    status: str
    version: str
    gemini_api_available: bool


# Initialize FastAPI app
app = FastAPI(
    title="Text-to-JSON Conversion API",
    description="Convert unstructured text into structured JSON following desired schemas",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
llm_manager = None
schema_analyzer = SchemaAnalyzer()
text_processor = TextProcessor()

# Initialize LLM manager
def initialize_llm():
    global llm_manager
    try:
        config = LLMConfig(
            api_key=os.getenv('GEMINI_API_KEY'),
            temperature=0.0,  # Use 0 temperature for consistency
            thinking_budget=30000  # Limited thinking for reliability
        )
        llm_manager = LLMManager(config)
        return True
    except Exception as e:
        print(f"Failed to initialize LLM: {e}")
        return False


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    success = initialize_llm()
    if not success:
        print("Warning: LLM Manager not initialized. Some features may not work.")


@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        gemini_api_available=llm_manager is not None
    )


@app.post("/convert", response_model=ConversionResponse)
async def convert_text_to_json(request: ConversionRequest):
    """
    Convert unstructured text to structured JSON following a schema.
    
    Args:
        request: ConversionRequest with input text, schema, and optional format
        
    Returns:
        ConversionResponse with generated JSON and metadata
    """
    if llm_manager is None:
        raise HTTPException(
            status_code=503, 
            detail="LLM Manager not available. Please check API key configuration."
        )
    
    try:
        result = await llm_manager.convert_to_json(
            input_text=request.input_text,
            schema=request.schema,
            input_format=request.input_format
        )
        
        return ConversionResponse(
            success=True,
            json_output=result.json_output,
            confidence_score=result.confidence_score,
            processing_time=result.processing_time,
            api_calls_made=result.api_calls_made,
            validation_errors=result.validation_errors,
            metadata=result.metadata
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/convert-file")
async def convert_file_to_json(
    file: UploadFile = File(...),
    schema: str = Form(..., description="JSON schema as string")
):
    """
    Convert uploaded file to structured JSON following a schema.
    
    Args:
        file: Uploaded file (PDF, DOCX, CSV, TXT, etc.)
        schema: JSON schema as string
        
    Returns:
        ConversionResponse with generated JSON and metadata
    """
    if llm_manager is None:
        raise HTTPException(
            status_code=503, 
            detail="LLM Manager not available. Please check API key configuration."
        )
    
    try:
        # Parse schema
        schema_dict = json.loads(schema)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON schema")
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Detect file type
        file_extension = Path(file.filename or "").suffix.lower().lstrip('.')
        
        # Process file
        processed_text = text_processor.process_input(
            input_data=file_content,
            file_type=file_extension,
            filename=file.filename
        )
        
        # Convert to JSON
        result = await llm_manager.convert_to_json(
            input_text=processed_text.content,
            schema=schema_dict,
            input_format=processed_text.format_type
        )
        
        # Add file processing metadata
        result.metadata["file_info"] = {
            "filename": file.filename,
            "size": len(file_content),
            "detected_type": file_extension,
            "processing_confidence": processed_text.confidence
        }
        
        return ConversionResponse(
            success=True,
            json_output=result.json_output,
            confidence_score=result.confidence_score,
            processing_time=result.processing_time,
            api_calls_made=result.api_calls_made,
            validation_errors=result.validation_errors,
            metadata=result.metadata
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-schema", response_model=SchemaAnalysisResponse)
async def analyze_schema(request: SchemaAnalysisRequest):
    """
    Analyze JSON schema complexity and determine processing strategy.
    
    Args:
        request: SchemaAnalysisRequest with JSON schema
        
    Returns:
        SchemaAnalysisResponse with metrics and strategy
    """
    try:
        metrics, strategy = schema_analyzer.analyze_schema(request.schema)
        
        # Estimate token usage for a sample text
        sample_text = "This is a sample text for token estimation."
        token_usage = schema_analyzer.estimate_token_usage(sample_text)
        
        return SchemaAnalysisResponse(
            metrics={
                "total_fields": metrics.total_fields,
                "nested_objects": metrics.nested_objects,
                "max_nesting_depth": metrics.max_nesting_depth,
                "enum_fields": metrics.enum_fields,
                "total_enum_values": metrics.total_enum_values,
                "required_fields": metrics.required_fields,
                "conditional_fields": metrics.conditional_fields,
                "complexity_score": metrics.complexity_score,
                "complexity_level": metrics.complexity_level.value
            },
            strategy={
                "use_multi_pass": strategy.use_multi_pass,
                "estimated_calls": strategy.estimated_calls,
                "chunk_schema": strategy.chunk_schema,
                "use_validation_loop": strategy.use_validation_loop,
                "max_retries": strategy.max_retries,
                "confidence_threshold": strategy.confidence_threshold
            },
            estimated_token_usage=token_usage
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/test-cases")
async def get_test_cases():
    """
    Get available test cases from the sample test cases directory.
    
    Returns:
        List of available test cases with their descriptions
    """
    test_cases_dir = Path("sample test cases")
    
    if not test_cases_dir.exists():
        return {"test_cases": []}
    
    test_cases = []
    
    for file_path in test_cases_dir.glob("*.json"):
        if "schema" in file_path.name:
            try:
                with open(file_path, 'r') as f:
                    schema = json.load(f)
                
                # Analyze the schema
                metrics, strategy = schema_analyzer.analyze_schema(schema)
                
                test_cases.append({
                    "name": file_path.stem,
                    "filename": file_path.name,
                    "complexity": metrics.complexity_level.value,
                    "fields": metrics.total_fields,
                    "nesting_depth": metrics.max_nesting_depth,
                    "enum_values": metrics.total_enum_values,
                    "estimated_calls": strategy.estimated_calls
                })
                
            except Exception as e:
                test_cases.append({
                    "name": file_path.stem,
                    "filename": file_path.name,
                    "error": str(e)
                })
    
    return {"test_cases": test_cases}


@app.get("/test-cases/{test_case_name}")
async def get_test_case(test_case_name: str):
    """
    Get a specific test case schema and any associated sample inputs.
    
    Args:
        test_case_name: Name of the test case
        
    Returns:
        Test case data including schema and sample inputs
    """
    test_cases_dir = Path("sample test cases")
    
    # Find schema file
    schema_file = None
    for file_path in test_cases_dir.glob(f"*{test_case_name}*.json"):
        if "schema" in file_path.name.lower():
            schema_file = file_path
            break
    
    if not schema_file:
        raise HTTPException(status_code=404, detail="Test case not found")
    
    try:
        with open(schema_file, 'r') as f:
            schema = json.load(f)
        
        # Look for associated sample inputs
        sample_inputs = []
        for file_path in test_cases_dir.glob("*"):
            if (test_case_name.lower() in file_path.name.lower() and 
                "schema" not in file_path.name.lower() and
                file_path.suffix in ['.md', '.txt', '.bib', '.csv']):
                
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    sample_inputs.append({
                        "filename": file_path.name,
                        "type": file_path.suffix.lstrip('.'),
                        "content": content[:1000] + "..." if len(content) > 1000 else content
                    })
                except Exception:
                    pass
        
        # Analyze schema
        metrics, strategy = schema_analyzer.analyze_schema(schema)
        
        return {
            "name": test_case_name,
            "schema": schema,
            "sample_inputs": sample_inputs,
            "analysis": {
                "complexity": metrics.complexity_level.value,
                "fields": metrics.total_fields,
                "nesting_depth": metrics.max_nesting_depth,
                "enum_values": metrics.total_enum_values,
                "estimated_calls": strategy.estimated_calls
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/test/{test_case_name}")
async def run_test_case(test_case_name: str, input_text: str = Form(...)):
    """
    Run a specific test case with provided input text.
    
    Args:
        test_case_name: Name of the test case
        input_text: Input text to convert
        
    Returns:
        ConversionResponse with results
    """
    if llm_manager is None:
        raise HTTPException(
            status_code=503, 
            detail="LLM Manager not available. Please check API key configuration."
        )
    
    # Get test case schema
    test_case_data = await get_test_case(test_case_name)
    schema = test_case_data["schema"]
    
    try:
        result = await llm_manager.convert_to_json(
            input_text=input_text,
            schema=schema,
            input_format="text"
        )
        
        result.metadata["test_case"] = test_case_name
        result.metadata["test_analysis"] = test_case_data["analysis"]
        
        return ConversionResponse(
            success=True,
            json_output=result.json_output,
            confidence_score=result.confidence_score,
            processing_time=result.processing_time,
            api_calls_made=result.api_calls_made,
            validation_errors=result.validation_errors,
            metadata=result.metadata
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_system_metrics():
    """
    Get system metrics and performance statistics.
    
    Returns:
        System metrics including API status and usage stats
    """
    return {
        "system_status": "operational",
        "llm_available": llm_manager is not None,
        "supported_formats": ["text", "pdf", "docx", "csv", "xlsx", "json"],
        "max_context_tokens": 128000,
        "max_output_tokens": 8192,
        "api_version": "1.0.0"
    }


if __name__ == "__main__":
    # Run the FastAPI application
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True
    ) 