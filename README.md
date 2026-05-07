# YouTube AI Chatbot — Chrome Extension

A production-ready Chrome Extension that lets you have real-time, intelligent conversations with any YouTube video. Powered by a FastAPI backend with a LangChain/LangGraph RAG pipeline and Pinecone vector database.

## Features

- 🎯 **Summarize** — Get key points from any length video
- 🔍 **Search** — Find specific moments with clickable timestamps
- 💡 **Clarify** — Get explanations of terms and concepts
- 🔗 **Contextualize** — Connect ideas across the video
- ⏱️ **Timestamp Navigation** — Click any timestamp to jump to that moment
- 🧠 **Conversation Memory** — Remembers your previous questions in the session
- 🛡️ **Grounded Answers** — Only answers based on the actual transcript

## Architecture

```
┌─────────────────────────┐     ┌──────────────────────────────────┐
│  Chrome Extension       │     │  FastAPI Backend                 │
│  (React + Tailwind)     │────▶│                                  │
│                         │     │  ┌─────────────────────────┐     │
│  • Side Panel UI        │     │  │  LangGraph Agent        │     │
│  • Content Script       │     │  │  ┌──────┐  ┌─────────┐  │     │
│  • Background Worker    │     │  │  │Route │→│Retrieve │  │     │
│                         │     │  │  └──────┘  └─────────┘  │     │
└─────────────────────────┘     │  │       ↓        ↓       │     │
                                │  │  ┌────────┐ ┌────────┐  │     │
                                │  │  │Generate│ │Guardrail│ │     │
                                │  │  └────────┘ └────────┘  │     │
                                │  └─────────────────────────┘     │
                                │         ↕                        │
                                │  ┌─────────────┐                 │
                                │  │  Pinecone   │                 │
                                │  └─────────────┘                 │
                                └──────────────────────────────────┘
```

## Quick Start

### 1. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys (OPENAI_API_KEY, PINECONE_API_KEY, etc.)

# Run the server
uvicorn app.main:app --reload --port 8000
```

### 2. Extension Setup

```bash
cd extension

# Install dependencies
npm install

# Build the extension
npm run build
```

### 3. Load Extension in Chrome

1. Open `chrome://extensions/`
2. Enable **Developer mode** (toggle in top-right)
3. Click **Load unpacked**
4. Select the `extension/dist` folder
5. Navigate to any YouTube video
6. Click the extension icon to open the side panel

## API Keys Required

| Service | Key | Purpose |
|---------|-----|---------|
| OpenAI | `OPENAI_API_KEY` | LLM (GPT-4o-mini) + Embeddings |
| Pinecone | `PINECONE_API_KEY` | Vector database |
| Cohere | `COHERE_API_KEY` | Reranking (optional) |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/ingest` | Index a YouTube video |
| `GET` | `/api/ingest/status/{video_id}` | Check indexing status |
| `POST` | `/api/chat` | Send a chat message |
| `DELETE` | `/api/chat/session/{session_id}` | Clear session |

## Tech Stack

- **Extension**: React 18, TypeScript, Tailwind CSS, Chrome MV3 Side Panel API
- **Backend**: FastAPI, Python 3.11+
- **RAG**: LangChain, LangGraph, Pinecone, Cohere Rerank
- **LLM**: OpenAI GPT-4o-mini
