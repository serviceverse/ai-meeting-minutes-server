import os
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import FileChatMessageHistory
from dotenv import load_dotenv
import json
import time
import threading
from pathlib import Path
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

HISTORY_DIR = Path("./chat_histories")
HISTORY_DIR.mkdir(parents=True, exist_ok=True)
FIRST_RUN_MARKER = HISTORY_DIR / ".first_run_complete"

# Global tracker for session files and their deletion times
# Format: {session_file_path: deletion_timestamp}
SESSION_TRACKER = {}

# Define OpenAI JSON Schema for Itinerary Planning
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "generate_itinerary",
            "description": "Generate a detailed travel itinerary based on user requirements. Create day-by-day plans with activities, places to visit, restaurants, and recommendations. Format the response in Markdown with proper structure including day headers, time periods (Morning, Afternoon, Evening), and place links using [[Place Name|place_id]] format.",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {
                        "type": "string",
                        "description": "The destination city or location for the trip"
                    },
                    "duration": {
                        "type": "string",
                        "description": "Duration of the trip (e.g., '3 days', '1 week', 'weekend')"
                    },
                    "interests": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of interests or activities the user wants to experience (e.g., 'museums', 'food', 'nature', 'nightlife')"
                    },
                    "budget": {
                        "type": "string",
                        "description": "Budget range or preference (e.g., 'budget', 'mid-range', 'luxury')"
                    },
                    "itineraryContent": {
                        "type": "string",
                        "description": "The complete itinerary content in Markdown format with day-by-day breakdown, including morning, afternoon, and evening activities. Use [[Place Name|place_id]] format for place links."
                    },
                    "additionalNotes": {
                        "type": "string",
                        "description": "Any additional tips, recommendations, or important information for the traveler"
                    }
                },
                "required": ["destination", "duration", "itineraryContent"]
            }
        }
    }
]

def get_session_history(session_id: str):
    """Get chat history for a session"""
    storage_path = "./chat_histories"
    if not os.path.exists(storage_path):
        os.makedirs(storage_path)
    
    file_path = os.path.join(storage_path, f"{session_id}.json")

    current_time = time.time()
    deletion_time = current_time + (1 * 3600)  # 1 hour from now
    SESSION_TRACKER[str(file_path)] = deletion_time
    return FileChatMessageHistory(file_path=file_path)




def cleanup_old_sessions(max_age_hours: int = 1, first_run_cleanup_all: bool = True):
    """
    Cleanup old session files from the chat_history directory based on global tracker.
    
    Args:
        max_age_hours: Maximum age in hours for session files (default: 1 hour)
        first_run_cleanup_all: If True, delete all files on first run (default: True)
    """
    if not HISTORY_DIR.exists():
        return
    
    current_time = time.time()
    deleted_count = 0
    
    # Check if this is the first run
    is_first_run = not FIRST_RUN_MARKER.exists()
    
    try:
        if is_first_run and first_run_cleanup_all:
            # First run: delete all session files and clear tracker
            session_files = list(HISTORY_DIR.glob("*.json"))
            print(f"[CLEANUP] First run detected. Cleaning up all {len(session_files)} session files...")
            
            for file_path in session_files:
                try:
                    file_path.unlink()
                    deleted_count += 1
                except Exception as e:
                    print(f"[CLEANUP] Error deleting {file_path}: {e}")
            
            # Clear the global tracker
            SESSION_TRACKER.clear()
            
            # Create marker file to indicate first run is complete
            try:
                FIRST_RUN_MARKER.touch()
                print(f"[CLEANUP] First run cleanup complete. Deleted {deleted_count} files. Tracker cleared.")
            except Exception as e:
                print(f"[CLEANUP] Error creating marker file: {e}")
        else:
            # Regular cleanup: check global tracker and delete files whose time has come
            files_to_delete = []
            
            for file_path_str, deletion_time in list(SESSION_TRACKER.items()):
                if current_time >= deletion_time:
                    files_to_delete.append(file_path_str)
            
            # Delete files whose deletion time has passed
            for file_path_str in files_to_delete:
                try:
                    file_path = Path(file_path_str)
                    if file_path.exists():
                        file_path.unlink()
                        deleted_count += 1
                        print(f"[CLEANUP] Deleted session file: {file_path.name} (deletion time reached)")
                    
                    # Remove from tracker
                    SESSION_TRACKER.pop(file_path_str, None)
                except Exception as e:
                    print(f"[CLEANUP] Error deleting {file_path_str}: {e}")
                    # Remove from tracker even if deletion failed
                    SESSION_TRACKER.pop(file_path_str, None)
            
            if deleted_count > 0:
                print(f"[CLEANUP] Cleanup complete. Deleted {deleted_count} old session file(s).")
    
    except Exception as e:
        print(f"[CLEANUP] Error during cleanup: {e}")

def cleanup_worker(interval_minutes: int = 5):
    """
    Background worker that periodically checks SESSION_TRACKER and runs cleanup.
    
    Args:
        interval_minutes: Interval in minutes between cleanup checks (default: 5 minutes)
    """
    # Run initial cleanup on first start
    cleanup_old_sessions()
    
    # Then run cleanup at regular intervals
    while True:
        time.sleep(interval_minutes * 60)  # Convert minutes to seconds
        cleanup_old_sessions(first_run_cleanup_all=False)

# Start cleanup thread in background
_cleanup_thread = threading.Thread(
    target=cleanup_worker,
    args=(5,),  # Check every 5 minutes
    daemon=True,  # Thread will exit when main program exits
    name="SessionCleanupThread"
)
_cleanup_thread.start()


def invoke_agent(session_id: str, system_prompt: str, message: str) -> str:
    """
    Invokes the agent with tool calling using OpenAI function calling.
    Returns the response text directly for chat-based itinerary planning.
    """
    config = {"configurable": {"session_id": session_id}}
    
    # Initialize LLM with tool binding
    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=api_key,
        temperature=0.7,
    ).bind(tools=TOOLS_SCHEMA)
    
    # Get session history
    history = get_session_history(session_id)
    messages = list(history.messages)
    
    # Add system prompt and user message
    current_messages = [
        SystemMessage(content=system_prompt),
        *messages,
        HumanMessage(content=message)
    ]
    
    print(f"\n[AGENT] Processing user request: {message}")
    
    # LLM call to get tool calls with content
    response = llm.invoke(current_messages)
    
    # Extract response text for fallback
    response_text = response.content if hasattr(response, 'content') else ""
    
    # Check if the response contains tool calls
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f"[AGENT] Tool calls detected: {len(response.tool_calls)}")
        
        # Process each tool call - extract itinerary content directly from arguments
        for tool_call in response.tool_calls:
            tool_name = tool_call.get('name')
            tool_args = tool_call.get('args', {})
            
            print(f"[AGENT] Executing tool: {tool_name}")
            
            # Get the itinerary content directly from tool call arguments
            itinerary_content = tool_args.get("itineraryContent", "")
            additional_notes = tool_args.get("additionalNotes", "")
            
            # Combine itinerary content with additional notes
            if itinerary_content:
                if additional_notes:
                    final_response_text = f"{itinerary_content}\n\n---\n\n**Additional Notes:**\n{additional_notes}"
                else:
                    final_response_text = itinerary_content
                
                # Save conversation to history
                history.add_message(HumanMessage(content=message))
                history.add_message(AIMessage(content=final_response_text))
                
                print(f"[AGENT] ✓ Successfully generated itinerary")
                return json.dumps({
                    "success": True,
                    "message": final_response_text
                })
        
        # If tool calls exist but no valid content, fall through to text response
        print(f"[AGENT] Tool calls found but no valid content, using response text")
    
    # No tool calls or fallback: return regular response text
    print(f"[AGENT] Returning conversational response")
    
    # Save to history
    history.add_message(HumanMessage(content=message))
    history.add_message(AIMessage(content=response_text))
    
    return json.dumps({
        "success": True,
        "message": response_text
    })


# Define OpenAI JSON Schema for Meeting Minutes Processing
MEETING_MINUTES_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "extract_meeting_minutes",
            "description": "Extract structured meeting minutes from transcript or text. Generate summary, minutes of meeting, key decisions, and action items.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "A concise executive summary of the meeting (2-3 sentences)"
                    },
                    "mom": {
                        "type": "string",
                        "description": "Complete minutes of meeting in Markdown format with agenda items, discussions, and decisions"
                    },
                    "keyDecisions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of key decisions made during the meeting"
                    },
                    "actionItems": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "task": {"type": "string", "description": "The action item task description"},
                                "assignee": {"type": "string", "description": "Person responsible for the task"},
                                "deadline": {"type": "string", "description": "Deadline in YYYY-MM-DD format"},
                                "priority": {"type": "string", "description": "Priority level: High, Medium, or Low"}
                            },
                            "required": ["task", "assignee", "deadline", "priority"]
                        },
                        "description": "List of action items with task, assignee, deadline, and priority"
                    },
                    "metadata": {
                        "type": "object",
                        "properties": {
                            "meetingDate": {"type": "string", "description": "Meeting date in ISO format"},
                            "durationSeconds": {"type": "integer", "description": "Meeting duration in seconds"},
                            "participants": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of meeting participants"
                            }
                        },
                        "description": "Meeting metadata including date, duration, and participants"
                    }
                },
                "required": ["summary", "mom", "keyDecisions", "actionItems"]
            }
        }
    }
]


def process_meeting_minutes(transcript: str, options: dict = None) -> dict:
    """
    Process meeting transcript and extract structured meeting minutes.
    
    Args:
        transcript: The meeting transcript or text content
        options: Optional processing options
        
    Returns:
        Dictionary containing summary, mom, keyDecisions, actionItems, and metadata
    """
    if options is None:
        options = {}
    
    # Initialize LLM with meeting minutes tool binding
    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=api_key,
        temperature=0.3,  # Lower temperature for more consistent extraction
    ).bind(tools=MEETING_MINUTES_TOOLS_SCHEMA)
    
    system_prompt = """You are an expert meeting minutes analyst. Your role is to analyze meeting transcripts and extract:

1. **Executive Summary**: A concise 2-3 sentence overview of the meeting
2. **Minutes of Meeting (MoM)**: Well-structured Markdown document with:
   - Agenda items as headers (### Agenda Item X: Title)
   - Discussion points under each agenda item
   - Decisions clearly marked with **Decision**: prefix
   - Action items mentioned in discussions
3. **Key Decisions**: A list of all important decisions made during the meeting
4. **Action Items**: Structured list with:
   - Task description
   - Assignee (extract from context, use "TBD" if unclear)
   - Deadline (extract from context, use reasonable estimate if unclear)
   - Priority (High/Medium/Low based on urgency and importance)
5. **Metadata**: Extract meeting date, duration, and participants if available

Be thorough and accurate. If information is not available in the transcript, make reasonable inferences or mark as "TBD" where appropriate."""
    
    user_message = f"""Please analyze the following meeting transcript and extract the meeting minutes:

{transcript}

Extract all relevant information including summary, detailed minutes, key decisions, and action items."""
    
    print(f"\n[AGENT] Processing meeting minutes from transcript (length: {len(transcript)} chars)")
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ]
    
    # LLM call to get structured extraction
    response = llm.invoke(messages)
    
    # Extract response
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f"[AGENT] Tool calls detected: {len(response.tool_calls)}")
        
        for tool_call in response.tool_calls:
            tool_name = tool_call.get('name')
            tool_args = tool_call.get('args', {})
            
            if tool_name == "extract_meeting_minutes":
                print(f"[AGENT] ✓ Successfully extracted meeting minutes")
                
                # Generate UUIDs for action items
                import uuid
                action_items = tool_args.get("actionItems", [])
                for item in action_items:
                    if "id" not in item:
                        item["id"] = str(uuid.uuid4())
                
                # Prepare metadata with defaults if not provided
                metadata = tool_args.get("metadata", {})
                if "meetingDate" not in metadata:
                    metadata["meetingDate"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                if "durationSeconds" not in metadata:
                    metadata["durationSeconds"] = 0
                if "participants" not in metadata:
                    metadata["participants"] = []
                
                return {
                    "summary": tool_args.get("summary", ""),
                    "mom": tool_args.get("mom", ""),
                    "keyDecisions": tool_args.get("keyDecisions", []),
                    "actionItems": action_items,
                    "metadata": metadata
                }
    
    # Fallback if tool call fails
    print(f"[AGENT] Warning: Tool call failed, using fallback extraction")
    return {
        "summary": "Meeting transcript processed. Please review the content.",
        "mom": transcript,
        "keyDecisions": [],
        "actionItems": [],
        "metadata": {
            "meetingDate": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "durationSeconds": 0,
            "participants": []
        }
    }


def answer_meeting_question(meeting_context: dict, question: str, session_id: str = None) -> str:
    """
    Answer questions about meeting results using the meeting context.
    
    Args:
        meeting_context: Dictionary containing meeting results (summary, mom, keyDecisions, actionItems, metadata)
        question: User's question about the meeting
        session_id: Optional session ID for conversation history
        
    Returns:
        JSON string with success and message fields
    """
    # Initialize LLM for Q&A
    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=api_key,
        temperature=0.7,  # Higher temperature for more natural conversation
    )
    
    # Build context from meeting results
    context_text = f"""Meeting Summary:
{meeting_context.get('summary', 'N/A')}

Key Decisions:
{chr(10).join('- ' + d for d in meeting_context.get('keyDecisions', []))}

Action Items:
{chr(10).join(f"- {item.get('task', 'N/A')} (Assignee: {item.get('assignee', 'TBD')}, Deadline: {item.get('deadline', 'TBD')}, Priority: {item.get('priority', 'TBD')})" for item in meeting_context.get('actionItems', []))}

Meeting Minutes:
{meeting_context.get('mom', 'N/A')[:2000]}  # Limit to avoid token limits

Meeting Metadata:
- Date: {meeting_context.get('metadata', {}).get('meetingDate', 'N/A')}
- Duration: {meeting_context.get('metadata', {}).get('durationSeconds', 0)} seconds
- Participants: {', '.join(meeting_context.get('metadata', {}).get('participants', []))}
"""
    
    system_prompt = """You are an AI assistant that helps users understand meeting minutes and results. Your role is to:

1. Answer questions about the meeting content, decisions, action items, and participants
2. Provide specific information from the meeting minutes when available
3. Be concise and accurate - only use information from the provided meeting context
4. If information is not available in the context, clearly state that
5. Help users understand timelines, responsibilities, and next steps
6. Be conversational and helpful

Always base your answers on the meeting context provided. Do not make up information."""
    
    user_message = f"""Based on the following meeting information, please answer this question:

{question}

Meeting Context:
{context_text}"""
    
    # Get session history if session_id provided
    if session_id:
        history = get_session_history(session_id)
        messages = list(history.messages)
    else:
        messages = []
    
    # Build message list
    current_messages = [
        SystemMessage(content=system_prompt),
        *messages,
        HumanMessage(content=user_message)
    ]
    
    print(f"\n[AGENT] Answering question about meeting: {question[:100]}")
    
    # Get response
    response = llm.invoke(current_messages)
    response_text = response.content if hasattr(response, 'content') else ""
    
    # Save to history if session_id provided
    if session_id:
        history.add_message(HumanMessage(content=question))
        history.add_message(AIMessage(content=response_text))
    
    print(f"[AGENT] ✓ Generated answer")
    
    return json.dumps({
        "success": True,
        "message": response_text
    })