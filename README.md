# Meeting Minutes Action Item Generator

This project is an AI-powered service designed to automatically extract meeting minutes, key decisions, and action items from meeting transcripts, audio, or video files. Leveraging OpenAI's GPT models and the LangChain framework, it processes meeting content and generates structured outputs including executive summaries, detailed minutes, key decisions, and actionable items with assignees and deadlines.

## Features

- **Multi-Format Support**: Process text transcripts, audio files, and video recordings.
- **Structured Extraction**: Automatically extracts executive summaries, meeting minutes, key decisions, and action items.
- **Action Item Tracking**: Generates action items with assignees, deadlines, and priority levels.
- **API Ready**: Built with FastAPI, making it easy to integrate with frontend applications.
- **Background Processing**: Asynchronous job processing with status tracking.
- **Session Management**: Built-in chat history tracking for coherent multi-turn interactions.

---

## Setup & Installation

### 1. Create a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

#### **Windows**
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate
```

#### **macOS / Linux**
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate
```

### 2. Install Dependencies

Once the virtual environment is activated, install the required packages:

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a file named `.env` in the root of the `backend` directory and add your OpenAI API key:

```env
OPENAI_API_KEY=your_api_key_here
```

---

## How to Run

### Start the API Server

You can run the server using the provided `main.py` script:

```bash
python main.py
```
The server will start at `http://localhost:8000`.

**Note**: The frontend is configured to use `http://localhost:3000/api/v1`. You may need to:
- Update the frontend `BASE_URL` in `frontend/src/services/api.js` to `http://localhost:8000/api/v1`, or
- Configure a proxy in your frontend development server to forward `/api/v1` requests to `http://localhost:8000`

### API Endpoints

#### Meeting Minutes Processing

- **POST `/api/v1/upload`**: Upload a media file (audio, video, or text document).
  - Content-Type: `multipart/form-data`
  - Body: Form data with `file` field
  - Response: `{"success": true, "data": {"uploadId": "...", "filename": "...", "type": "...", "size": ..., "uploadedAt": "..."}}`

- **POST `/api/v1/process`**: Start processing an uploaded file.
  - Body: `{"uploadId": "upload-id-from-upload-endpoint", "options": {}}`
  - Response: `{"success": true, "data": {"jobId": "...", "status": "processing", "estimatedTimeSeconds": 30, "message": "..."}}`

- **GET `/api/v1/status/{jobId}`**: Check the status of a processing job.
  - Response: `{"success": true, "data": {"jobId": "...", "status": "processing|completed|failed", "progress": 0-100, "currentStage": "...", "updatedAt": "..."}}`

- **GET `/api/v1/results/{jobId}`**: Get the results of a completed job.
  - Response: `{"success": true, "data": {"jobId": "...", "metadata": {...}, "content": {"summary": "...", "mom": "...", "keyDecisions": [...], "actionItems": [...]}}}`

#### Chat API

- **POST `/api/chat`**: Chat endpoint for conversational interactions.
  - Body: `{"message": "Your message", "sessionId": "unique-session-id"}`
  - Response: `{"success": true, "message": "AI response"}`

#### Health Check

- **GET `/health`**: Check if the service is running.
  - Response: `{"status": "healthy", "service": "Meeting Minutes Action Item Generator"}`

---

## Testing

You can use the provided test scripts to verify the functionality:
- `python test.py`: Runs automated API tests.
- `python interactive_test.py`: Interactive CLI for testing the agent.
