"""
FastAPI web server for Cloud Run deployment.

Provides HTTP endpoints for:
- Health checks
- PDF upload and processing
- Results retrieval
"""

import os
import json
import logging
import tempfile
import re
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import pipeline components
from ehrx.vlm.pipeline import DocumentPipeline
from ehrx.vlm.grouping import SubDocumentGrouper, generate_hierarchical_index
from ehrx.vlm.config import VLMConfig
from ehrx.agent.query import HybridQueryAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="EHRX Pipeline API",
    description="EHR extraction pipeline using Google Gemini VLM",
    version="1.0.0"
)

# Configure CORS (allow override via env variable, fallback to known frontend)
default_origins = [
    "https://frontend-795204058658.europe-west1.run.app",
    "http://localhost:5173",
    "http://localhost:4173"
]
allowed_origins = os.getenv("CORS_ALLOW_ORIGINS")
if allowed_origins:
    origins_list = [origin.strip() for origin in allowed_origins.split(",") if origin.strip()]
else:
    origins_list = default_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
OUTPUT_DIR = Path("/tmp/ehrx_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_DOCS_DIR = Path(os.getenv("SAMPLE_DOCS_DIR", "./SampleEHR_docs")).resolve()
SAMPLE_DOCS_DIR.mkdir(parents=True, exist_ok=True)

# Precomputed cache directory (checked into repo and copied into image)
PRECOMPUTED_DIR = Path(os.getenv("PRECOMPUTED_DIR", "./precomputed_samples")).resolve()
PRECOMPUTED_DIR.mkdir(parents=True, exist_ok=True)

# Precomputed answers directory (checked into repo and copied into image)
PRECOMPUTED_ANSWERS_DIR = Path(os.getenv("PRECOMPUTED_ANSWERS_DIR", "./precomputed_answers")).resolve()
PRECOMPUTED_ANSWERS_DIR.mkdir(parents=True, exist_ok=True)

# Precompute controls
PRECOMPUTE_SAMPLES = os.getenv("PRECOMPUTE_SAMPLES", "false").lower() in {"1", "true", "yes", "y"}

# In-memory caches for precomputed samples
# filename -> stable document_id
PRECOMPUTED_SAMPLE_MAP: dict[str, str] = {}
# document_id -> enhanced ontology JSON
IN_MEMORY_ONTOLOGY: dict[str, dict] = {}
# Set of known precomputed document_ids
PRECOMPUTED_DOC_IDS: set[str] = set()

def _discover_sample_documents() -> list[dict]:
    docs = []
    if not SAMPLE_DOCS_DIR.exists():
        logger.warning(f"Sample docs dir not found: {SAMPLE_DOCS_DIR}")
        return docs
    for pdf in sorted(SAMPLE_DOCS_DIR.glob("*.pdf")):
        docs.append({
            "id": pdf.name,
            "filename": pdf.name,
            "display_name": pdf.stem.replace("_", " "),
            "size_bytes": pdf.stat().st_size
        })
    return docs

SAMPLE_DOCS_CACHE = _discover_sample_documents()


def _ensure_sample_document(filename: str) -> Path:
    for doc in SAMPLE_DOCS_CACHE:
        if doc["filename"] == filename:
            return SAMPLE_DOCS_DIR / filename
    raise HTTPException(status_code=404, detail=f"Sample document {filename} not found")

# Global pipeline instance (initialized lazily)
vlm_config = None
pipeline = None
query_agent = None


def get_pipeline():
    """Get or create pipeline instance."""
    global vlm_config, pipeline
    if pipeline is None:
        vlm_config = VLMConfig.from_env()
        pipeline = DocumentPipeline(
            vlm_config=vlm_config,
            checkpoint_interval=50,
            dpi=200
        )
        logger.info("Pipeline initialized")
    return pipeline


class QueryBody(BaseModel):
    """Request model for querying results."""
    question: str


def _get_document_paths(document_id: str) -> dict:
    """Return common file paths for a document."""
    doc_dir = OUTPUT_DIR / document_id
    return {
        "dir": doc_dir,
        "enhanced": doc_dir / f"{document_id}_enhanced.json",
        "index": doc_dir / f"{document_id}_index.json",
        "full": doc_dir / f"{document_id}_full.json"
    }


def _process_pdf_file(
    pdf_path: str,
    original_filename: str,
    page_range: str,
    document_type: str,
    *,
    document_id_override: Optional[str] = None
) -> dict:
    """Shared PDF processing routine."""
    # Document ID should be just the base filename without extension, no numbers appended
    # Also clean any _artifacts suffix or trailing _digits
    if document_id_override:
        document_id = document_id_override
    else:
        raw_stem = Path(original_filename).stem
        document_id = _clean_document_id(raw_stem)
    doc_paths = _get_document_paths(document_id)
    doc_paths["dir"].mkdir(parents=True, exist_ok=True)

    pipeline_instance = get_pipeline()
    document = pipeline_instance.process_document(
        pdf_path=pdf_path,
        output_dir=str(doc_paths["dir"]),
        page_range=page_range,
        document_context={"document_type": document_type}
    )

    grouper = SubDocumentGrouper(confidence_threshold=0.80)
    enhanced_doc = grouper.group_document(document)

    with open(doc_paths["enhanced"], "w") as f:
        json.dump(enhanced_doc, f, indent=2)

    index = generate_hierarchical_index(enhanced_doc)
    with open(doc_paths["index"], "w") as f:
        json.dump(index, f, indent=2)

    return {
        "document_id": document_id,
        "document": document,
        "enhanced": enhanced_doc
    }


def _slugify_filename(filename: str) -> str:
    base = Path(filename).stem.lower()
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in base)
    while "--" in safe:
        safe = safe.replace("--", "-")
    return safe.strip("-_") or "sample"


def precompute_sample_documents() -> None:
    """
    Precompute enhanced ontology for bundled sample PDFs (once), store on disk,
    and load the enhanced JSON into memory cache for fast retrieval.
    """
    if not SAMPLE_DOCS_DIR.exists():
        logger.warning(f"Sample docs directory missing: {SAMPLE_DOCS_DIR}")
        return

    logger.info("Precomputing sample documents (if needed)...")
    for entry in SAMPLE_DOCS_DIR.glob("*.pdf"):
        try:
            slug = _slugify_filename(entry.name)
            stable_doc_id = f"sample_{slug}"
            paths = _get_document_paths(stable_doc_id)

            if paths["enhanced"].exists():
                # Already computed on disk; load into memory
                with open(paths["enhanced"], "r") as f:
                    IN_MEMORY_ONTOLOGY[stable_doc_id] = json.load(f)
                PRECOMPUTED_SAMPLE_MAP[entry.name] = stable_doc_id
                logger.info(f"Loaded precomputed: {entry.name} -> {stable_doc_id}")
                continue

            # Not computed yet; run pipeline once
            logger.info(f"Precomputing: {entry} -> {stable_doc_id}")
            result = _process_pdf_file(
                pdf_path=str(entry),
                original_filename=entry.name,
                page_range="all",
                document_type="Clinical EHR",
                document_id_override=stable_doc_id
            )
            IN_MEMORY_ONTOLOGY[stable_doc_id] = result["enhanced"]
            PRECOMPUTED_SAMPLE_MAP[entry.name] = stable_doc_id
            logger.info(f"Precomputed and cached: {entry.name} -> {stable_doc_id}")
        except Exception as e:
            logger.error(f"Failed to precompute {entry.name}: {e}", exc_info=True)

def _clean_document_id(filename_stem: str) -> str:
    """
    Extract clean document ID from filename stem.
    Removes '_artifacts' suffix and any trailing underscore + numeric digits.
    Examples:
      'Complete History and Physical Write-Up_artifacts' -> 'Complete History and Physical Write-Up'
      'Document_123' -> 'Document'
      'Document_artifacts_456' -> 'Document'
    """
    # Remove _artifacts suffix if present
    if filename_stem.endswith("_artifacts"):
        filename_stem = filename_stem[:-10]  # len("_artifacts") == 10
    
    # Remove trailing underscore + digits pattern (e.g., "_123", "_456")
    filename_stem = re.sub(r'_\d+$', '', filename_stem)
    
    return filename_stem


def load_precomputed_cache_from_repo() -> None:
    """
    Load precomputed samples from PRECOMPUTED_DIR into memory and filename mapping.
    Supports current canonical single-file layout and legacy files:
      Canonical (preferred):
        precomputed_samples/<doc_id>/<doc_id>.json
        precomputed_samples/<doc_id>/metadata.json
      Flat (supported):
        precomputed_samples/<doc_id>.json
        precomputed_samples/<doc_id>_artifacts.json  (cleaned to <doc_id>)
      Legacy fallbacks:
        precomputed_samples/<doc_id>/<doc_id>_enhanced.json
        precomputed_samples/<doc_id>/<doc_id>_index.json
    """
    if not PRECOMPUTED_DIR.exists():
        return
    # Load flat files at root: precomputed_samples/<doc_id>.json or <doc_id>_artifacts.json
    for flat_json in PRECOMPUTED_DIR.glob("*.json"):
        try:
            # Extract clean document ID (remove _artifacts suffix and trailing _digits)
            raw_stem = flat_json.stem
            doc_id = _clean_document_id(raw_stem)
            
            with open(flat_json, "r") as f:
                enhanced_json = json.load(f)
            IN_MEMORY_ONTOLOGY[doc_id] = enhanced_json
            PRECOMPUTED_DOC_IDS.add(doc_id)
            # Map PDF filename to document ID (PDF should match doc_id + ".pdf")
            PRECOMPUTED_SAMPLE_MAP[f"{doc_id}.pdf"] = doc_id
            # Mirror into OUTPUT_DIR for listing parity
            out_paths = _get_document_paths(doc_id)
            out_paths["dir"].mkdir(parents=True, exist_ok=True)
            with open(out_paths["enhanced"], "w") as f:
                json.dump(enhanced_json, f, indent=2)
            logger.info(f"Loaded precomputed: {flat_json.name} -> doc_id={doc_id}")
        except Exception as e:
            logger.warning(f"Failed loading precomputed flat file {flat_json}: {e}")
    for doc_dir in PRECOMPUTED_DIR.iterdir():
        if not doc_dir.is_dir():
            continue
        doc_id = doc_dir.name
        # Preferred single-file artifact
        canonical = doc_dir / f"{doc_id}.json"
        # Legacy alternatives
        enhanced = doc_dir / f"{doc_id}_enhanced.json"
        index_fallback = doc_dir / f"{doc_id}_index.json"
        metadata = doc_dir / "metadata.json"
        # Determine which JSON to load
        chosen_json = None
        if canonical.exists():
            chosen_json = canonical
        elif enhanced.exists():
            chosen_json = enhanced
        elif index_fallback.exists():
            chosen_json = index_fallback
        if chosen_json is None or not metadata.exists():
            continue
        try:
            with open(chosen_json, "r") as f:
                enhanced_json = json.load(f)
            with open(metadata, "r") as f:
                meta = json.load(f)
            IN_MEMORY_ONTOLOGY[doc_id] = enhanced_json
            PRECOMPUTED_DOC_IDS.add(doc_id)
            orig = meta.get("original_filename")
            if orig:
                PRECOMPUTED_SAMPLE_MAP[orig] = doc_id
            # Also copy to OUTPUT_DIR to make /documents listing work uniformly
            out_paths = _get_document_paths(doc_id)
            out_paths["dir"].mkdir(parents=True, exist_ok=True)
            with open(out_paths["enhanced"], "w") as f:
                json.dump(enhanced_json, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed loading precomputed {doc_dir}: {e}")

async def precompute_samples_background() -> None:
    """Run precompute in a background thread to avoid blocking startup."""
    await asyncio.to_thread(precompute_sample_documents)

@app.on_event("startup")
async def on_startup():
    # Always load pre-shipped precomputed cache if present
    load_precomputed_cache_from_repo()
    # Schedule precompute in the background if enabled (non-blocking)
    if PRECOMPUTE_SAMPLES:
        try:
            asyncio.create_task(precompute_samples_background())
            logger.info("Scheduled background precompute for sample documents")
        except Exception as e:
            logger.warning(f"Failed to schedule background precompute: {e}")


@app.get("/")
async def root():
    """Root endpoint - API information."""
    return {
        "service": "EHRX Pipeline API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "upload": "/upload (POST)",
            "sample_documents": "/sample-documents (GET)",
            "sample_preview": "/sample-documents/{filename} (GET)",
            "process_sample": "/sample-documents/{filename}/process (POST)",
            "ontology": "/documents/{id}/ontology (GET)",
            "query": "/documents/{id}/query (POST)",
            "precomputed_answers_list": "/precomputed-answers (GET)",
            "precomputed_answers_get": "/precomputed-answers/{document_id} (GET)",
            "precomputed_answers_alt": "/documents/{document_id}/precomputed-answers (GET)",
            "docs": "/docs"
        },
        "notes": "Upload PDFs directly to /upload using multipart/form-data"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for Cloud Run."""
    try:
        # Check if VLM config is accessible
        config = VLMConfig.from_env()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "project_id": config.project_id,
            "model": config.model_name
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    page_range: str = "all",
    document_type: str = "Clinical EHR"
):
    """
    Upload and process a PDF through the VLM extraction pipeline.
    
    Args:
        file: PDF file to process
        page_range: Page range to process (e.g., "1-10", "all")
        document_type: Type of document for context
    
    Returns:
        Processing job information with document_id
    """
    logger.info(f"Received PDF: {file.filename}, page_range: {page_range}")
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    try:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_pdf_path = tmp_file.name
        
        logger.info(f"Saved PDF to: {tmp_pdf_path}")
        
        result = _process_pdf_file(
            pdf_path=tmp_pdf_path,
            original_filename=file.filename,
            page_range=page_range,
            document_type=document_type
        )
        os.unlink(tmp_pdf_path)

        document = result["document"]
        total_pages = document.get("total_pages") or document["processing_stats"].get("total_pages")
        status = "complete"
        
        return {
            "document_id": result["document_id"],
            "status": status,
            "total_pages": total_pages,
            "enhanced_json_url": f"/documents/{result['document_id']}/ontology"
        }
        
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.get("/sample-documents")
async def list_sample_documents():
    """Return available bundled sample PDFs, with precomputed IDs when available."""
    samples = []
    for item in SAMPLE_DOCS_CACHE:
        filename = item["filename"]
        doc_id = PRECOMPUTED_SAMPLE_MAP.get(filename)
        samples.append({
            **item,
            "precomputed": bool(doc_id),
            "document_id": doc_id
        })
    return {"samples": samples}


@app.get("/sample-documents/{filename}")
async def get_sample_document(filename: str):
    """
    Serve a bundled sample PDF file for preview/download.
    
    Args:
        filename: Name of the sample PDF file
    
    Returns:
        PDF file response
    """
    sample_path = _ensure_sample_document(filename)
    
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail=f"Sample document {filename} not found")
    
    return FileResponse(
        sample_path,
        media_type="application/pdf",
        filename=filename,
        headers={
            "Content-Disposition": f'inline; filename="{filename}"'
        }
    )


@app.post("/sample-documents/{filename}/process")
async def process_sample_document(
    filename: str,
    page_range: str = "all",
    document_type: str = "Clinical EHR"
):
    """
    Return precomputed result for a bundled sample PDF immediately (no processing delay).
    Uses cached JSON files from precomputed_samples/ directory.
    Falls back to on-demand processing only if not precomputed.
    """
    sample_path = _ensure_sample_document(filename)
    logger.info(f"Requested sample processing: {sample_path}")

    # Extract clean document ID from filename (remove .pdf, _artifacts, trailing _digits)
    raw_stem = Path(filename).stem
    stable_doc_id = _clean_document_id(raw_stem)
    
    # Check if we have this in precomputed cache (by filename or by cleaned doc_id)
    if filename in PRECOMPUTED_SAMPLE_MAP:
        stable_doc_id = PRECOMPUTED_SAMPLE_MAP[filename]
    elif stable_doc_id in PRECOMPUTED_DOC_IDS:
        # Found by cleaned doc_id, update mapping
        PRECOMPUTED_SAMPLE_MAP[filename] = stable_doc_id
    # Check if we have precomputed data (immediate return, no delay)
    if stable_doc_id in IN_MEMORY_ONTOLOGY:
        enhanced = IN_MEMORY_ONTOLOGY[stable_doc_id]
        total_pages = enhanced.get("total_pages") or enhanced.get("processing_stats", {}).get("total_pages", 0)
        logger.info(f"Returning precomputed result immediately: {filename} -> {stable_doc_id}")
        return {
            "document_id": stable_doc_id,
            "status": "complete",
            "total_pages": total_pages,
            "enhanced_json_url": f"/documents/{stable_doc_id}/ontology",
            "source": "sample_document_precomputed",
            "filename": filename
        }
    
    # Try to load from precomputed files on disk (check various naming patterns)
    precomputed_patterns = [
        PRECOMPUTED_DIR / f"{stable_doc_id}.json",
        PRECOMPUTED_DIR / f"{stable_doc_id}_artifacts.json",
        PRECOMPUTED_DIR / stable_doc_id / f"{stable_doc_id}.json",
        PRECOMPUTED_DIR / stable_doc_id / f"{stable_doc_id}_enhanced.json",
        PRECOMPUTED_DIR / stable_doc_id / f"{stable_doc_id}_index.json",
    ]
    
    for json_path in precomputed_patterns:
        if json_path.exists():
            try:
                with open(json_path, "r") as f:
                    enhanced = json.load(f)
                IN_MEMORY_ONTOLOGY[stable_doc_id] = enhanced
                PRECOMPUTED_DOC_IDS.add(stable_doc_id)
                PRECOMPUTED_SAMPLE_MAP[filename] = stable_doc_id
                # Mirror to OUTPUT_DIR for listing
                out_paths = _get_document_paths(stable_doc_id)
                out_paths["dir"].mkdir(parents=True, exist_ok=True)
                with open(out_paths["enhanced"], "w") as f:
                    json.dump(enhanced, f, indent=2)
                total_pages = enhanced.get("total_pages") or enhanced.get("processing_stats", {}).get("total_pages", 0)
                logger.info(f"Loaded and returning precomputed: {filename} -> {stable_doc_id}")
                return {
                    "document_id": stable_doc_id,
                    "status": "complete",
                    "total_pages": total_pages,
                    "enhanced_json_url": f"/documents/{stable_doc_id}/ontology",
                    "source": "sample_document_precomputed",
                    "filename": filename
                }
            except Exception as e:
                logger.warning(f"Failed to load {json_path}: {e}")
                continue

    # Fallback: process on demand and cache
    result = _process_pdf_file(
        pdf_path=str(sample_path),
        original_filename=filename,
        page_range=page_range,
        document_type=document_type
    )
    document = result["document"]
    total_pages = document.get("total_pages") or document.get("processing_stats", {}).get("total_pages")
    return {
        "document_id": result["document_id"],
        "status": "complete",
        "total_pages": total_pages,
        "enhanced_json_url": f"/documents/{result['document_id']}/ontology",
        "source": "sample_document_processed_now",
        "filename": filename
    }


@app.get("/documents/{document_id}/ontology")
async def get_ontology(document_id: str):
    """Return enhanced ontology JSON for a document."""
    paths = _get_document_paths(document_id)
    # Prefer in-memory cache immediately
    cached = IN_MEMORY_ONTOLOGY.get(document_id)
    if cached is not None:
        return cached
    # Look in precomputed dir (check various naming patterns including _artifacts.json)
    precomputed_patterns = [
        PRECOMPUTED_DIR / f"{document_id}.json",
        PRECOMPUTED_DIR / f"{document_id}_artifacts.json",
        PRECOMPUTED_DIR / document_id / f"{document_id}.json",
        PRECOMPUTED_DIR / document_id / f"{document_id}_enhanced.json",
        PRECOMPUTED_DIR / document_id / f"{document_id}_index.json",
    ]
    chosen = None
    for pattern in precomputed_patterns:
        if pattern.exists():
            chosen = pattern
            break
    if chosen and chosen.exists():
        try:
            with open(chosen, "r") as f:
                enhanced = json.load(f)
            IN_MEMORY_ONTOLOGY[document_id] = enhanced
            # Mirror into OUTPUT_DIR for listing parity
            out_paths = _get_document_paths(document_id)
            out_paths["dir"].mkdir(parents=True, exist_ok=True)
            with open(out_paths["enhanced"], "w") as f:
                json.dump(enhanced, f, indent=2)
            return enhanced
        except Exception as e:
            logger.error(f"Failed to load precomputed ontology for {document_id}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to load ontology")
    # Otherwise try runtime output dir (legacy enhanced path)
    if not paths["enhanced"].exists():
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    
    try:
        with open(paths["enhanced"], "r") as f:
            enhanced = json.load(f)
        # Populate cache for subsequent requests
        IN_MEMORY_ONTOLOGY[document_id] = enhanced
        return enhanced
    except Exception as e:
        logger.error(f"Failed to load ontology for {document_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to load ontology")


@app.post("/documents/{document_id}/query")
async def query_document(document_id: str, request: QueryBody):
    """
    Query a processed document using natural language.
    
    Returns:
        Query results with matched elements and answer
    """
    logger.info(f"Query for document {document_id}: {request.question}")
    
    # Find the enhanced document
    enhanced_path = _get_document_paths(document_id)["enhanced"]
    
    if not enhanced_path.exists():
        raise HTTPException(
            status_code=404, 
            detail=f"Document {document_id} not found. Upload a PDF first."
        )
    
    try:
        # Initialize query agent if needed
        global query_agent, vlm_config
        
        if vlm_config is None:
            vlm_config = VLMConfig.from_env()
        
        # Create agent for this query
        agent = HybridQueryAgent(schema_path=str(enhanced_path), vlm_config=vlm_config)
        
        # Execute query
        result = agent.query(request.question)
        
        logger.info(f"Query returned {len(result['matched_elements'])} matches")
        
        matched_elements = []
        for element in result.get("matched_elements", []):
            matched_elements.append({
                "element_id": element.get("element_id"),
                "type": element.get("type"),
                "content": element.get("content"),
                "page_number": element.get("page_number"),
                "bbox_pixel": element.get("bbox_pixel"),
                "bbox_pdf": element.get("bbox_pdf"),
                "relevance": element.get("relevance")
            })
        
        return {
            "answer_summary": result.get("answer_summary") or result.get("answer", ""),
            "matched_elements": matched_elements,
            "reasoning": result.get("reasoning", ""),
            "filter_stats": result.get("filter_stats", {})
        }
        
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/precomputed-answers")
async def list_precomputed_answers():
    """
    List all available precomputed answer files.
    
    Returns:
        List of documents with precomputed answers, including metadata
    """
    answers = []
    
    if not PRECOMPUTED_ANSWERS_DIR.exists():
        logger.warning(f"Precomputed answers directory not found: {PRECOMPUTED_ANSWERS_DIR}")
        return {
            "status": "success",
            "total_documents": 0,
            "documents": []
        }
    
    # Find all _answers.json files
    for answer_file in sorted(PRECOMPUTED_ANSWERS_DIR.glob("*_answers.json")):
        try:
            with open(answer_file, 'r') as f:
                data = json.load(f)
                # Extract document_id from filename (remove _answers.json suffix)
                doc_id = answer_file.stem.replace("_answers", "")
                answers.append({
                    "document_id": doc_id,
                    "filename": answer_file.name,
                    "total_questions": data.get("total_questions", 0),
                    "source_json": data.get("source_json", ""),
                    "answers_url": f"/precomputed-answers/{doc_id}"
                })
        except Exception as e:
            logger.warning(f"Error reading precomputed answers file {answer_file.name}: {e}")
    
    return {
        "status": "success",
        "total_documents": len(answers),
        "documents": answers
    }


@app.get("/precomputed-answers/{document_id}")
async def get_precomputed_answers(document_id: str):
    """
    Get precomputed answers for a specific document.
    
    Args:
        document_id: The document ID (filename without .pdf extension)
    
    Returns:
        Complete precomputed answers JSON with all questions, answers, and matched elements
    """
    # Try multiple filename patterns
    possible_files = [
        PRECOMPUTED_ANSWERS_DIR / f"{document_id}_answers.json",
        PRECOMPUTED_ANSWERS_DIR / f"{document_id}.json",  # Fallback if naming differs
    ]
    
    answer_file = None
    for path in possible_files:
        if path.exists():
            answer_file = path
            break
    
    if not answer_file or not answer_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Precomputed answers not found for document: {document_id}"
        )
    
    try:
        with open(answer_file, 'r') as f:
            answers_data = json.load(f)
        return answers_data
    except Exception as e:
        logger.error(f"Failed to load precomputed answers for {document_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load precomputed answers: {str(e)}"
        )


@app.get("/documents/{document_id}/precomputed-answers")
async def get_document_precomputed_answers(document_id: str):
    """
    Alternative endpoint to get precomputed answers for a document.
    This is a convenience endpoint that mirrors /precomputed-answers/{document_id}.
    """
    return await get_precomputed_answers(document_id)


@app.get("/documents")
async def list_documents():
    """List all processed documents."""
    documents = []
    
    if OUTPUT_DIR.exists():
        for doc_dir in OUTPUT_DIR.iterdir():
            if doc_dir.is_dir():
                enhanced_file = doc_dir / f"{doc_dir.name}_enhanced.json"
                if enhanced_file.exists():
                    try:
                        with open(enhanced_file, 'r') as f:
                            data = json.load(f)
                            documents.append({
                                "document_id": doc_dir.name,
                                "total_pages": data.get('total_pages', 0),
                                "sub_documents": len(data.get('sub_documents', [])),
                                "results_url": f"/results/{doc_dir.name}"
                            })
                    except Exception as e:
                        logger.warning(f"Error reading {doc_dir.name}: {e}")
    
    return {
        "status": "success",
        "total_documents": len(documents),
        "documents": documents
    }


# Cloud Run entry point
if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable (Cloud Run sets this)
    port = int(os.environ.get("PORT", 8080))
    
    logger.info(f"Starting EHRX Pipeline API on port {port}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )

