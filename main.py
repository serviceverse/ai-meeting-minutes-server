from fastapi import FastAPI, HTTPException, Request, UploadFile, File, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import invoke_agent, process_meeting_minutes, answer_meeting_question
from models import (
    ChatRequest, ChatResponse, UploadResponse, ProcessRequest, ProcessResponse,
    StatusResponse, ResultsResponse, MeetingChatRequest, MeetingChatResponse
)
import os
from dotenv import load_dotenv
import json
import uuid
import time
import threading
from pathlib import Path
from typing import Dict, Optional, Union

# Load environment variables
load_dotenv()

app = FastAPI(title="Meeting Minutes Action Item Generator API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def cleanup_chat_histories_folder():
    """
    Check if chat_histories folder exists and delete all files inside it.
    This function runs on app startup.
    """
    chat_histories_path = Path("./chat_histories")
    
    if chat_histories_path.exists() and chat_histories_path.is_dir():
        try:
            # Get all files in the directory
            files = list(chat_histories_path.glob("*"))
            deleted_count = 0
            
            for file_path in files:
                # Only delete files, not directories
                if file_path.is_file():
                    try:
                        file_path.unlink()
                        deleted_count += 1
                        print(f"[STARTUP] Deleted chat history file: {file_path.name}")
                    except Exception as e:
                        print(f"[STARTUP] Error deleting {file_path.name}: {e}")
            
            if deleted_count > 0:
                print(f"[STARTUP] Cleaned up {deleted_count} file(s) from chat_histories folder")
            else:
                print(f"[STARTUP] chat_histories folder is empty, no files to delete")
        except Exception as e:
            print(f"[STARTUP] Error cleaning chat_histories folder: {e}")
    else:
        print(f"[STARTUP] chat_histories folder does not exist, skipping cleanup")

@app.on_event("startup")
async def startup_event():
    """Run cleanup on app startup"""
    cleanup_chat_histories_folder()
# Job storage (in-memory with file persistence)
JOBS_STORAGE: Dict[str, dict] = {}
UPLOADS_STORAGE: Dict[str, dict] = {}

# Create directories
UPLOADS_DIR = Path("./uploads")
UPLOADS_DIR.mkdir(exist_ok=True)
JOBS_DIR = Path("./jobs")
JOBS_DIR.mkdir(exist_ok=True)

def load_jobs_from_disk():
    """Load jobs from disk on startup"""
    try:
        for job_file in JOBS_DIR.glob("*.json"):
            try:
                with open(job_file, "r", encoding="utf-8") as f:
                    job_data = json.load(f)
                    job_id = job_data.get("jobId")
                    if job_id:
                        JOBS_STORAGE[job_id] = job_data
                        print(f"[STORAGE] Loaded job {job_id} from disk")
            except Exception as e:
                print(f"[STORAGE] Error loading job file {job_file}: {e}")
    except Exception as e:
        print(f"[STORAGE] Error loading jobs: {e}")

def save_job_to_disk(job_id: str, job_data: dict):
    """Save job to disk for persistence"""
    try:
        job_file = JOBS_DIR / f"{job_id}.json"
        with open(job_file, "w", encoding="utf-8") as f:
            json.dump(job_data, f, indent=2, default=str)
    except Exception as e:
        print(f"[STORAGE] Error saving job {job_id} to disk: {e}")

def delete_job_from_disk(job_id: str):
    """Delete job file from disk"""
    try:
        job_file = JOBS_DIR / f"{job_id}.json"
        if job_file.exists():
            job_file.unlink()
    except Exception as e:
        print(f"[STORAGE] Error deleting job {job_id} from disk: {e}")

# Load existing jobs on startup
load_jobs_from_disk()
print(f"[STORAGE] Loaded {len(JOBS_STORAGE)} jobs from disk")

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint for itinerary planning.
    
    Request body:
    - message: String containing user's message/query
    - sessionId: String for session tracking
    
    Response:
    - success: Boolean indicating if response was generated successfully
    - message: String containing the AI's response
    """
    try:
        # Check if OPENAI_API_KEY is set
        if not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not found in environment variables.")

        # Create system prompt for the itinerary planning agent
        system_prompt = """You are an expert travel planner and itinerary builder. Your role is to help users plan amazing trips by:

1. Understanding their travel preferences, interests, and requirements
2. Creating detailed, day-by-day itineraries with specific activities, places to visit, and recommendations
3. Formatting responses in Markdown with clear structure:
   - Use ### for day headers (e.g., ### Day 1 - City Name)
   - Use #### for time periods (e.g., #### ☀️ Morning, #### 🌤️ Afternoon, #### 🌙 Evening)
   - Use **bold** for important place names
   - Use [[Place Name|place_id]] format for place links (e.g., [[Rijksmuseum|place_rijks]])
   - Include practical tips, restaurant recommendations, and cultural insights
4. Being conversational and helpful, asking clarifying questions when needed
5. Remembering context from previous messages in the conversation

Always provide detailed, actionable itineraries that help travelers make the most of their trips."""

        # Invoke agent with the user message
        response_json = invoke_agent(
            session_id=request.sessionId,
            system_prompt=system_prompt,
            message=request.message
        )

        print(f"[MAIN] Response: {response_json}")
        
        # Parse and return the JSON response
        response_data = json.loads(response_json)
        return ChatResponse(
            success=response_data.get("success", True),
            message=response_data.get("message", "")
        )
        
    except Exception as e:
        print(f"[MAIN] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Meeting Minutes API Endpoints

@app.post("/api/v1/upload", response_model=UploadResponse)
async def upload_media(file: UploadFile = File(...)):
    """
    Upload media file (audio, video, or text document).
    
    Returns:
    - success: Boolean indicating if upload was successful
    - data: Object containing uploadId, filename, type, size, uploadedAt
    """
    print(f"\n{'='*80}")
    print(f"[UPLOAD] ===== File Upload Request =====")
    print(f"[UPLOAD] Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[UPLOAD] File name: {file.filename}")
    print(f"[UPLOAD] Content type: {file.content_type}")
    
    try:
        # Generate unique upload ID
        upload_id = str(uuid.uuid4())
        print(f"[UPLOAD] Generated upload ID: {upload_id}")
        
        # Save file to uploads directory
        file_path = UPLOADS_DIR / f"{upload_id}_{file.filename}"
        print(f"[UPLOAD] Saving file to: {file_path}")
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        file_size = len(content)
        print(f"[UPLOAD] ✓ File saved successfully: {file_size} bytes")
        print(f"[UPLOAD] File exists: {file_path.exists()}")
        
        # Store upload metadata
        upload_data = {
            "uploadId": upload_id,
            "filename": file.filename,
            "type": file.content_type or "unknown",
            "size": file_size,
            "uploadedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "filePath": str(file_path)
        }
        UPLOADS_STORAGE[upload_id] = upload_data
        print(f"[UPLOAD] ✓ Upload metadata stored")
        print(f"[UPLOAD] Total uploads in storage: {len(UPLOADS_STORAGE)}")
        
        print(f"[UPLOAD] ✓ File uploaded successfully: {file.filename} (ID: {upload_id})")
        print(f"{'='*80}\n")
        
        return UploadResponse(
            success=True,
            data=upload_data
        )
    except Exception as e:
        print(f"[UPLOAD] ERROR: Exception occurred")
        print(f"[UPLOAD] Exception type: {type(e).__name__}")
        print(f"[UPLOAD] Exception message: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"{'='*80}\n")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/api/v1/process", response_model=ProcessResponse)
async def start_processing(request: ProcessRequest, background_tasks: BackgroundTasks):
    """
    Start processing uploaded media to extract meeting minutes.
    
    Request body:
    - uploadId: String ID from upload endpoint
    - options: Optional processing options
    
    Returns:
    - success: Boolean indicating if processing started successfully
    - data: Object containing jobId, status, estimatedTimeSeconds, message
    """
    try:
        # Check if upload exists
        if request.uploadId not in UPLOADS_STORAGE:
            raise HTTPException(status_code=404, detail="Upload not found")
        
        upload_data = UPLOADS_STORAGE[request.uploadId]
        file_path = Path(upload_data["filePath"])
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Initialize job status
        job_data = {
            "jobId": job_id,
            "uploadId": request.uploadId,
            "status": "processing",
            "progress": 0,
            "startTime": time.time(),
            "options": request.options,
            "result": None
        }
        JOBS_STORAGE[job_id] = job_data
        save_job_to_disk(job_id, job_data)  # Persist immediately
        
        # Start background processing
        background_tasks.add_task(process_job, job_id, file_path, upload_data)
        
        print(f"[PROCESS] Started processing job: {job_id} (Total jobs: {len(JOBS_STORAGE)})")
        
        return ProcessResponse(
            success=True,
            data={
                "jobId": job_id,
                "status": "processing",
                "estimatedTimeSeconds": 30,  # Estimate based on file size
                "message": "Processing started successfully."
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"[PROCESS] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


def process_job(job_id: str, file_path: Path, upload_data: dict):
    """Background job processor (synchronous function for BackgroundTasks)"""
    print(f"\n{'='*80}")
    print(f"[JOB {job_id}] ===== Background Job Processing Started =====")
    print(f"[JOB {job_id}] Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[JOB {job_id}] File path: {file_path}")
    print(f"[JOB {job_id}] Upload data: {upload_data.get('filename', 'N/A')} ({upload_data.get('type', 'N/A')})")
    
    try:
        job = JOBS_STORAGE[job_id]
        print(f"[JOB {job_id}] Job found in storage")
        
        # Update progress: Reading file
        print(f"[JOB {job_id}] Step 1/4: Reading file...")
        job["progress"] = 10
        job["status"] = "processing"
        print(f"[JOB {job_id}] Progress: 10%")
        
        # Read file content
        # For now, we'll handle text files. Audio/video would need transcription service
        content = ""
        if upload_data["type"].startswith("text/"):
            print(f"[JOB {job_id}] Reading text file...")
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            print(f"[JOB {job_id}] ✓ File read successfully: {len(content)} characters")
        else:
            # For audio/video files, we'd need a transcription service
            # For now, return an error or use a placeholder
            content = f"[Note: Audio/Video transcription not yet implemented. File: {upload_data['filename']}]"
            print(f"[JOB {job_id}] ⚠ Non-text file type, using placeholder")
        
        job["progress"] = 30
        print(f"[JOB {job_id}] Progress: 30%")
        
        # Process meeting minutes
        print(f"[JOB {job_id}] Step 2/4: Processing meeting minutes...")
        print(f"[JOB {job_id}] Calling process_meeting_minutes()...")
        result = process_meeting_minutes(content, job.get("options", {}))
        print(f"[JOB {job_id}] ✓ Meeting minutes processing completed")
        
        job["progress"] = 80
        print(f"[JOB {job_id}] Progress: 80%")
        
        # Store result
        print(f"[JOB {job_id}] Step 3/4: Storing results...")
        job["result"] = result
        job["progress"] = 100
        job["status"] = "completed"
        job["completedAt"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        print(f"[JOB {job_id}] ✓ Results stored in job data")
        
        # Log extracted information
        print(f"[JOB {job_id}] Extracted summary length: {len(result.get('summary', ''))} chars")
        print(f"[JOB {job_id}] Extracted MoM length: {len(result.get('mom', ''))} chars")
        print(f"[JOB {job_id}] Key decisions count: {len(result.get('keyDecisions', []))}")
        print(f"[JOB {job_id}] Action items count: {len(result.get('actionItems', []))}")
        
        # Persist to disk
        print(f"[JOB {job_id}] Step 4/4: Persisting to disk...")
        save_job_to_disk(job_id, job)
        print(f"[JOB {job_id}] ✓ Job persisted to disk")
        
        print(f"[JOB {job_id}] ✓ Processing completed successfully")
        print(f"[JOB {job_id}] Final status: {job['status']}, Progress: {job['progress']}%")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"[JOB {job_id}] ERROR: Exception occurred during processing")
        print(f"[JOB {job_id}] Exception type: {type(e).__name__}")
        print(f"[JOB {job_id}] Exception message: {str(e)}")
        import traceback
        print(f"[JOB {job_id}] Traceback:")
        traceback.print_exc()
        
        if job_id in JOBS_STORAGE:
            job = JOBS_STORAGE[job_id]
            job["status"] = "failed"
            job["error"] = str(e)
            save_job_to_disk(job_id, job)
            print(f"[JOB {job_id}] Job status updated to 'failed' and persisted")
        
        print(f"{'='*80}\n")


@app.get("/api/v1/status/{job_id}", response_model=StatusResponse)
async def check_status(job_id: str):
    """
    Check the status of a processing job.
    
    Returns:
    - success: Boolean indicating if job was found
    - data: Object containing jobId, status, progress, currentStage, updatedAt
    - error: Object with code and message if job not found
    """
    # Try to load from disk if not in memory
    if job_id not in JOBS_STORAGE:
        job_file = JOBS_DIR / f"{job_id}.json"
        if job_file.exists():
            try:
                with open(job_file, "r", encoding="utf-8") as f:
                    job_data = json.load(f)
                    JOBS_STORAGE[job_id] = job_data
            except Exception as e:
                print(f"[STATUS] Error loading job from disk: {e}")
                return StatusResponse(
                    success=False,
                    error={"code": "NOT_FOUND", "message": f"Job not found: {job_id}"}
                )
        else:
            return StatusResponse(
                success=False,
                error={"code": "NOT_FOUND", "message": f"Job not found: {job_id}"}
            )
    
    job = JOBS_STORAGE[job_id]
    
    # Determine current stage based on progress
    progress = job.get("progress", 0)
    if progress < 25:
        stage = "Uploading & Validating..."
    elif progress < 50:
        stage = "Transcribing Audio..."
    elif progress < 75:
        stage = "Extracting Key Points..."
    else:
        stage = "Finalizing Documents..."
    
    return StatusResponse(
        success=True,
        data={
            "jobId": job_id,
            "status": job.get("status", "unknown"),
            "progress": progress,
            "currentStage": stage,
            "updatedAt": time.strftime("%Y-%m-%dT%H:%M:%S")
        }
    )


@app.post("/api/v1/process-direct", response_model=ResultsResponse)
async def process_direct(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None)
):
    """
    Direct processing endpoint that accepts file or text and returns results immediately.
    
    Request:
    - file: Optional file upload (multipart/form-data)
    - text: Optional text content (form-data or JSON)
    
    Returns:
    - success: Boolean indicating if processing was successful
    - data: Object containing jobId, metadata, and content (summary, mom, keyDecisions, actionItems)
    """
    print(f"\n{'='*80}")
    print(f"[DIRECT] ===== Direct Processing Request Received =====")
    print(f"[DIRECT] Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[DIRECT] File provided: {file is not None}")
    print(f"[DIRECT] Text provided: {text is not None}")
    
    try:
        # Check if OPENAI_API_KEY is set
        print(f"[DIRECT] Checking OPENAI_API_KEY...")
        if not os.getenv("OPENAI_API_KEY"):
            print(f"[DIRECT] ERROR: OPENAI_API_KEY not found in environment variables")
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not found in environment variables.")
        print(f"[DIRECT] ✓ OPENAI_API_KEY found")
        
        content = ""
        filename = "direct_input"
        
        # Get content from file or text
        if file:
            print(f"[DIRECT] Processing file upload...")
            print(f"[DIRECT] File name: {file.filename}")
            print(f"[DIRECT] File content type: {file.content_type}")
            
            # Read file content
            content_bytes = await file.read()
            print(f"[DIRECT] File size: {len(content_bytes)} bytes")
            
            if file.content_type and file.content_type.startswith("text/"):
                content = content_bytes.decode("utf-8")
                print(f"[DIRECT] ✓ Decoded text file successfully")
            else:
                # For non-text files, we'd need transcription
                content = f"[Note: Audio/Video transcription not yet implemented. File: {file.filename}]"
                print(f"[DIRECT] ⚠ Non-text file type, using placeholder")
            filename = file.filename or "uploaded_file"
            print(f"[DIRECT] Processing file: {filename} ({len(content)} chars)")
        elif text:
            print(f"[DIRECT] Processing text input...")
            content = text
            filename = "text_input"
            print(f"[DIRECT] Text length: {len(content)} chars")
            print(f"[DIRECT] Text preview (first 100 chars): {content[:100]}...")
        else:
            print(f"[DIRECT] ERROR: Neither file nor text provided")
            raise HTTPException(status_code=400, detail="Either 'file' or 'text' must be provided")
        
        if not content.strip():
            print(f"[DIRECT] ERROR: Content is empty after processing")
            raise HTTPException(status_code=400, detail="Content is empty")
        
        print(f"[DIRECT] ✓ Content validated: {len(content)} characters")
        
        # Process meeting minutes directly
        print(f"[DIRECT] Starting meeting minutes processing...")
        print(f"[DIRECT] Calling process_meeting_minutes()...")
        result = process_meeting_minutes(content, {})
        print(f"[DIRECT] ✓ Meeting minutes processing completed")
        
        # Log extracted information
        print(f"[DIRECT] Extracted summary length: {len(result.get('summary', ''))} chars")
        print(f"[DIRECT] Extracted MoM length: {len(result.get('mom', ''))} chars")
        print(f"[DIRECT] Key decisions count: {len(result.get('keyDecisions', []))}")
        print(f"[DIRECT] Action items count: {len(result.get('actionItems', []))}")
        
        # Generate a temporary job ID for the response
        job_id = str(uuid.uuid4())
        print(f"[DIRECT] Generated job ID: {job_id}")
        
        print(f"[DIRECT] ✓ Processing completed successfully")
        print(f"[DIRECT] Returning results...")
        print(f"{'='*80}\n")
        
        return ResultsResponse(
            success=True,
            data={
                "jobId": job_id,
                "metadata": result.get("metadata", {}),
                "content": {
                    "summary": result.get("summary", ""),
                    "mom": result.get("mom", ""),
                    "keyDecisions": result.get("keyDecisions", []),
                    "actionItems": result.get("actionItems", [])
                }
            }
        )
        
    except HTTPException as he:
        print(f"[DIRECT] HTTPException raised: {he.status_code} - {he.detail}")
        print(f"{'='*80}\n")
        raise
    except Exception as e:
        print(f"[DIRECT] ERROR: Unexpected exception occurred")
        print(f"[DIRECT] Exception type: {type(e).__name__}")
        print(f"[DIRECT] Exception message: {str(e)}")
        import traceback
        print(f"[DIRECT] Traceback:")
        traceback.print_exc()
        print(f"{'='*80}\n")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.get("/api/v1/results/{job_id}", response_model=ResultsResponse)
async def get_results(job_id: str):
    """
    Get the results of a completed processing job.
    
    Returns:
    - success: Boolean indicating if results were retrieved successfully
    - data: Object containing jobId, metadata, and content (summary, mom, keyDecisions, actionItems)
    """
    print(f"[RESULTS] Request for job: {job_id}")
    print(f"[RESULTS] Available jobs: {list(JOBS_STORAGE.keys())[:5]}...")  # Show first 5 for debugging
    
    # Try to load from disk if not in memory
    if job_id not in JOBS_STORAGE:
        job_file = JOBS_DIR / f"{job_id}.json"
        if job_file.exists():
            try:
                with open(job_file, "r", encoding="utf-8") as f:
                    job_data = json.load(f)
                    JOBS_STORAGE[job_id] = job_data
                    print(f"[RESULTS] Loaded job {job_id} from disk")
            except Exception as e:
                print(f"[RESULTS] Error loading job from disk: {e}")
                raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        else:
            print(f"[RESULTS] Job file not found: {job_file}")
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}. Available jobs: {len(JOBS_STORAGE)}")
    
    job = JOBS_STORAGE[job_id]
    
    if job.get("status") == "failed":
        error_msg = job.get("error", "Unknown error")
        raise HTTPException(status_code=500, detail=f"Job failed: {error_msg}")
    
    if job.get("status") != "completed":
        status = job.get("status", "unknown")
        progress = job.get("progress", 0)
        raise HTTPException(
            status_code=400, 
            detail=f"Job is not yet completed. Status: {status}, Progress: {progress}%"
        )
    
    result = job.get("result")
    if not result:
        raise HTTPException(status_code=500, detail="Results not available for this job")
    
    print(f"[RESULTS] Returning results for job {job_id}")
    return ResultsResponse(
        success=True,
        data={
            "jobId": job_id,
            "metadata": result.get("metadata", {}),
            "content": {
                "summary": result.get("summary", ""),
                "mom": result.get("mom", ""),
                "keyDecisions": result.get("keyDecisions", []),
                "actionItems": result.get("actionItems", [])
            }
        }
    )


# Chat API Endpoint (existing)
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint for itinerary planning.
    
    Request body:
    - message: String containing user's message/query
    - sessionId: String for session tracking
    
    Response:
    - success: Boolean indicating if response was generated successfully
    - message: String containing the AI's response
    """
    try:
        # Check if OPENAI_API_KEY is set
        if not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not found in environment variables.")

        # Create system prompt for the itinerary planning agent
        system_prompt = """You are an expert travel planner and itinerary builder. Your role is to help users plan amazing trips by:

1. Understanding their travel preferences, interests, and requirements
2. Creating detailed, day-by-day itineraries with specific activities, places to visit, and recommendations
3. Formatting responses in Markdown with clear structure:
   - Use ### for day headers (e.g., ### Day 1 - City Name)
   - Use #### for time periods (e.g., #### ☀️ Morning, #### 🌤️ Afternoon, #### 🌙 Evening)
   - Use **bold** for important place names
   - Use [[Place Name|place_id]] format for place links (e.g., [[Rijksmuseum|place_rijks]])
   - Include practical tips, restaurant recommendations, and cultural insights
4. Being conversational and helpful, asking clarifying questions when needed
5. Remembering context from previous messages in the conversation

Always provide detailed, actionable itineraries that help travelers make the most of their trips."""

        # Invoke agent with the user message
        response_json = invoke_agent(
            session_id=request.sessionId,
            system_prompt=system_prompt,
            message=request.message
        )

        print(f"[MAIN] Response: {response_json}")
        
        # Parse and return the JSON response
        response_data = json.loads(response_json)
        return ChatResponse(
            success=response_data.get("success", True),
            message=response_data.get("message", "")
        )
        
    except Exception as e:
        print(f"[MAIN] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/chat", response_model=MeetingChatResponse)
async def chat_about_meeting(request: MeetingChatRequest):
    """
    Chat endpoint for asking questions about meeting results.
    
    Request body:
    - jobId: String ID of the processed meeting job
    - message: String containing user's question about the meeting
    
    Response:
    - success: Boolean indicating if response was generated successfully
    - message: String containing the AI's response
    """
    try:
        # Check if job exists and get results (try loading from disk if not in memory)
        if request.jobId not in JOBS_STORAGE:
            job_file = JOBS_DIR / f"{request.jobId}.json"
            if job_file.exists():
                try:
                    with open(job_file, "r", encoding="utf-8") as f:
                        job_data = json.load(f)
                        JOBS_STORAGE[request.jobId] = job_data
                        print(f"[CHAT] Loaded job {request.jobId} from disk")
                except Exception as e:
                    print(f"[CHAT] Error loading job from disk: {e}")
                    raise HTTPException(status_code=404, detail=f"Job not found: {request.jobId}")
            else:
                raise HTTPException(status_code=404, detail=f"Job not found: {request.jobId}")
        
        job = JOBS_STORAGE[request.jobId]
        
        if job.get("status") != "completed":
            raise HTTPException(status_code=400, detail="Job is not yet completed")
        
        result = job.get("result")
        if not result:
            raise HTTPException(status_code=500, detail="Meeting results not available")
        
        # Check if OPENAI_API_KEY is set
        if not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not found in environment variables.")
        
        # Use jobId as session_id for conversation history
        session_id = f"meeting_chat_{request.jobId}"
        
        # Get meeting context from results
        meeting_context = {
            "summary": result.get("summary", ""),
            "mom": result.get("mom", ""),
            "keyDecisions": result.get("keyDecisions", []),
            "actionItems": result.get("actionItems", []),
            "metadata": result.get("metadata", {})
        }
        
        # Invoke agent to answer the question
        response_json = answer_meeting_question(
            meeting_context=meeting_context,
            question=request.message,
            session_id=session_id
        )
        
        print(f"[MAIN] Chat response generated for job {request.jobId}")
        
        # Parse and return the JSON response
        response_data = json.loads(response_json)
        return MeetingChatResponse(
            success=response_data.get("success", True),
            message=response_data.get("message", "")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[MAIN] Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Meeting Minutes Action Item Generator"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
