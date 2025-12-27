from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class ChatRequest(BaseModel):
    message: str  # User's chat message
    sessionId: str

class ChatResponse(BaseModel):
    success: bool
    message: str  # AI response message

# Meeting Minutes API Models
class UploadResponse(BaseModel):
    success: bool
    data: Dict[str, Any]  # Contains uploadId, filename, type, size, uploadedAt

class ProcessRequest(BaseModel):
    uploadId: str
    options: Optional[Dict[str, Any]] = {}

class ProcessResponse(BaseModel):
    success: bool
    data: Dict[str, Any]  # Contains jobId, status, estimatedTimeSeconds, message

class StatusResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None  # Contains jobId, status, progress, currentStage, updatedAt
    error: Optional[Dict[str, str]] = None  # Contains code and message if error

class ActionItem(BaseModel):
    id: str
    task: str
    assignee: str
    deadline: str
    priority: str

class MeetingMetadata(BaseModel):
    meetingDate: str
    durationSeconds: int
    participants: List[str]

class MeetingContent(BaseModel):
    summary: str
    mom: str  # Minutes of Meeting in Markdown
    keyDecisions: List[str]
    actionItems: List[ActionItem]

class ResultsResponse(BaseModel):
    success: bool
    data: Dict[str, Any]  # Contains jobId, metadata, content

class MeetingChatRequest(BaseModel):
    jobId: str
    message: str

class MeetingChatResponse(BaseModel):
    success: bool
    message: str  # AI response message

class DirectProcessRequest(BaseModel):
    text: Optional[str] = None  # Text content to process
    # File will be handled separately via UploadFile
