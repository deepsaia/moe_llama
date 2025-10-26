"""
Chat server with web UI and streaming API for MoE language model.

This FastAPI server provides:
- Streaming chat completions API
- Model management with multiple checkpoint support
- Session management (user sessions and chat sessions)
- Built-in web UI (React app served from frontend/dist)
- Health monitoring and statistics

Usage:
    python -m scripts.chat_server
    python -m scripts.chat_server --port 8080
    python -m scripts.chat_server --host 0.0.0.0

Endpoints:
    GET  /                       - Chat UI (React app)
    POST /api/chat/completions   - Streaming chat API
    GET  /api/models             - List available models
    GET  /api/health             - Health check
    GET  /api/stats              - Server statistics
"""

import argparse
import os
import json
import uuid
import time
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Optional, AsyncGenerator, Dict
from datetime import datetime

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from moellama import LLaMA4MoE, BPETokenizer, load_config, setup_device

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Request limits (abuse prevention)
MAX_MESSAGES_PER_REQUEST = 100
MAX_MESSAGE_LENGTH = 4000
MAX_TOTAL_CONVERSATION_LENGTH = 16000
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0
MIN_TOP_K = 1
MAX_TOP_K = 200
MIN_MAX_TOKENS = 1
MAX_MAX_TOKENS = 2048

# Parse arguments
parser = argparse.ArgumentParser(description='MoE Chat Server')
parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to bind')
parser.add_argument('--port', type=int, default=8000, help='Port to run on')
parser.add_argument('--config', type=str, default='config.hocon', help='Config file')
parser.add_argument('--model-dir', type=str, default=None, help='Override model directory')
parser.add_argument('--dev-mode', action='store_true', help='Development mode (CORS, etc.)')
args = parser.parse_args()


# ====================
# Data Models
# ====================

class ChatMessage(BaseModel):
    """A single chat message."""
    role: str  # 'user' or 'assistant'
    content: str


class ChatRequest(BaseModel):
    """Chat completion request."""
    messages: List[ChatMessage]
    model: Optional[str] = None  # Model name (optional)
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    session_id: Optional[str] = None  # For session tracking


class ModelInfo(BaseModel):
    """Information about an available model."""
    id: str
    name: str
    size: int
    modified: str


# ====================
# Model Manager
# ====================

class ModelManager:
    """
    Manages multiple model checkpoints.

    Handles loading, caching, and switching between different trained models.
    """

    def __init__(self, config_path: str, model_dir: Optional[str] = None):
        self.config = load_config(config_path)
        self.device = setup_device(self.config)

        # Model directory
        if model_dir:
            self.model_dir = Path(model_dir)
        else:
            self.model_dir = Path(self.config['paths']['model_path'])

        # Cache for loaded models
        self.models: Dict[str, tuple] = {}  # model_id -> (model, tokenizer)
        self.current_model_id: Optional[str] = None

        logger.info(f"ModelManager initialized. Device: {self.device}")
        logger.info(f"Model directory: {self.model_dir}")

    def list_models(self) -> List[ModelInfo]:
        """List all available model checkpoints."""
        if not self.model_dir.exists():
            logger.warning(f"Model directory does not exist: {self.model_dir}")
            return []

        models = []
        for model_file in self.model_dir.glob("model_*.pt"):
            stat = model_file.stat()
            models.append(ModelInfo(
                id=model_file.stem,
                name=model_file.name,
                size=stat.st_size,
                modified=datetime.fromtimestamp(stat.st_mtime).isoformat()
            ))

        # Sort by modification time (newest first)
        models.sort(key=lambda m: m.modified, reverse=True)
        return models

    def load_model(self, model_id: str) -> tuple:
        """
        Load a specific model checkpoint.

        Args:
            model_id: Model identifier (e.g., "model_tiny_shakespeare_20250126")

        Returns:
            Tuple of (model, tokenizer)
        """
        # Check cache
        if model_id in self.models:
            logger.info(f"Using cached model: {model_id}")
            return self.models[model_id]

        # Find model and vocab files
        model_file = self.model_dir / f"{model_id}.pt"
        if not model_file.exists():
            raise FileNotFoundError(f"Model not found: {model_file}")

        # Find corresponding vocab file
        # Extract dataset name from model_id
        # Pattern: model_{dataset}_{timestamp} or model_{dataset}_{date}_{time}
        # Example: model_tiny_shakespeare_20251026_070939
        parts = model_id.split('_')

        # Try to find the vocab file by removing timestamp parts
        # Timestamps are typically 8 digits (YYYYMMDD) or 6 digits (HHMMSS)
        dataset_parts = []
        for i, part in enumerate(parts):
            if i == 0:  # Skip "model"
                continue
            # If part looks like a timestamp (all digits, 6-8 chars), stop
            if part.isdigit() and len(part) >= 6:
                break
            dataset_parts.append(part)

        if dataset_parts:
            dataset_name = '_'.join(dataset_parts)
            vocab_file = self.model_dir / f"vocab_{dataset_name}.txt"
        else:
            # Fallback: find any vocab file
            vocab_files = list(self.model_dir.glob("vocab_*.txt"))
            if not vocab_files:
                raise FileNotFoundError(f"No vocab file found in {self.model_dir}")
            vocab_file = vocab_files[0]

        if not vocab_file.exists():
            # Second fallback: try to find a matching vocab file
            vocab_pattern = f"vocab_{dataset_name if dataset_parts else '*'}.txt"
            vocab_files = list(self.model_dir.glob(vocab_pattern))
            if vocab_files:
                vocab_file = vocab_files[0]
            else:
                raise FileNotFoundError(f"Vocab file not found: {vocab_file}")

        logger.info(f"Loading model: {model_file}")
        logger.info(f"Loading vocab: {vocab_file}")

        # Load tokenizer
        tokenizer = BPETokenizer()
        tokenizer.load_vocab(str(vocab_file))
        vocab_size = len(tokenizer)

        # Create model
        model = LLaMA4MoE(
            vocab_size=vocab_size,
            dim=self.config['model']['dim'],
            num_layers=self.config['model']['num_layers'],
            num_heads=self.config['model']['num_heads'],
            num_experts=self.config['model']['num_experts'],
            top_k=self.config['model']['top_k'],
            max_seq_len=self.config['model']['max_seq_len'],
            dropout=self.config['model']['dropout'],
            shared_expert=self.config['model']['shared_expert'],
            load_balancing_loss_coef=self.config['model']['load_balancing_loss_coef']
        )

        # Load weights
        state_dict = torch.load(str(model_file), map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        # Cache
        self.models[model_id] = (model, tokenizer)
        self.current_model_id = model_id

        logger.info(f"Model loaded successfully: {model_id}")
        return model, tokenizer

    def get_default_model(self) -> tuple:
        """Load the most recent model checkpoint."""
        models = self.list_models()
        if not models:
            raise FileNotFoundError("No trained models found. Please train a model first.")

        # Load most recent
        model_id = models[0].id
        logger.info(f"Loading default (most recent) model: {model_id}")
        return self.load_model(model_id)


# ====================
# Request Validation
# ====================

def validate_chat_request(request: ChatRequest):
    """Validate chat request to prevent abuse."""
    if len(request.messages) == 0:
        raise HTTPException(status_code=400, detail="At least one message required")

    if len(request.messages) > MAX_MESSAGES_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"Too many messages. Max {MAX_MESSAGES_PER_REQUEST} allowed"
        )

    # Check message lengths
    total_length = 0
    for i, msg in enumerate(request.messages):
        if not msg.content:
            raise HTTPException(status_code=400, detail=f"Message {i} has empty content")

        msg_length = len(msg.content)
        if msg_length > MAX_MESSAGE_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Message {i} too long. Max {MAX_MESSAGE_LENGTH} chars"
            )
        total_length += msg_length

    if total_length > MAX_TOTAL_CONVERSATION_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Conversation too long. Max {MAX_TOTAL_CONVERSATION_LENGTH} chars"
        )

    # Validate roles
    for i, msg in enumerate(request.messages):
        if msg.role not in ["user", "assistant"]:
            raise HTTPException(
                status_code=400,
                detail=f"Message {i} has invalid role. Must be 'user' or 'assistant'"
            )

    # Validate parameters
    if request.temperature is not None:
        if not (MIN_TEMPERATURE <= request.temperature <= MAX_TEMPERATURE):
            raise HTTPException(
                status_code=400,
                detail=f"Temperature must be {MIN_TEMPERATURE}-{MAX_TEMPERATURE}"
            )

    if request.top_k is not None:
        if not (MIN_TOP_K <= request.top_k <= MAX_TOP_K):
            raise HTTPException(
                status_code=400,
                detail=f"top_k must be {MIN_TOP_K}-{MAX_TOP_K}"
            )

    if request.max_tokens is not None:
        if not (MIN_MAX_TOKENS <= request.max_tokens <= MAX_MAX_TOKENS):
            raise HTTPException(
                status_code=400,
                detail=f"max_tokens must be {MIN_MAX_TOKENS}-{MAX_MAX_TOKENS}"
            )


# ====================
# FastAPI App
# ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load default model on startup."""
    logger.info("Starting MoE Chat Server...")

    # Initialize model manager
    app.state.model_manager = ModelManager(args.config, args.model_dir)

    # Load default model
    try:
        model, tokenizer = app.state.model_manager.get_default_model()
        logger.info(f"Server ready at http://{args.host}:{args.port}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.warning("Server starting without a loaded model. Load via API.")

    # Session tracking
    app.state.sessions = {}  # session_id -> session_data

    yield

    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="MoE Chat API",
    description="Streaming chat API for Mixture of Experts language model",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if args.dev_mode else ["http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ====================
# API Endpoints
# ====================

@app.post("/api/chat/completions")
async def chat_completions(request: ChatRequest):
    """
    Streaming chat completions endpoint.

    Generates assistant response in a streaming fashion, yielding tokens
    as they are generated.
    """
    validate_chat_request(request)

    # Get model
    model_manager = app.state.model_manager
    if request.model:
        try:
            model, tokenizer = model_manager.load_model(request.model)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Model not found: {str(e)}")
    else:
        model, tokenizer = model_manager.get_default_model()

    # Get generation parameters
    temperature = request.temperature or app.state.model_manager.config['inference']['temperature']
    max_tokens = request.max_tokens or app.state.model_manager.config['inference']['max_new_tokens']
    top_k = request.top_k or app.state.model_manager.config['inference']['top_k']
    top_p = request.top_p or app.state.model_manager.config['inference']['top_p']

    # Format conversation
    conversation_text = ""
    for msg in request.messages:
        if msg.role == "user":
            conversation_text += f"User: {msg.content}\n"
        elif msg.role == "assistant":
            conversation_text += f"Assistant: {msg.content}\n"
    conversation_text += "Assistant: "

    logger.info(f"Chat request: {len(request.messages)} messages, temp={temperature}")

    async def generate_stream() -> AsyncGenerator[str, None]:
        """Generate response stream."""
        try:
            # Encode prompt
            token_ids = tokenizer.encode(conversation_text)
            input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
            input_ids = input_ids.to(model_manager.device)

            # Generate with streaming
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )

            # Decode
            generated_text = tokenizer.decode(generated_ids[0].tolist())
            response_text = generated_text[len(conversation_text):]

            # Stream response in chunks
            chunk_size = 5  # Characters per chunk
            for i in range(0, len(response_text), chunk_size):
                chunk = response_text[i:i+chunk_size]
                # SSE format
                yield f"data: {json.dumps({'content': chunk})}\n\n"

            # Send done signal
            yield f"data: {json.dumps({'done': True})}\n\n"

        except Exception as e:
            logger.exception("Error during generation")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/api/models")
async def list_models():
    """List all available models."""
    models = app.state.model_manager.list_models()
    return {
        "models": [model.model_dump() for model in models],
        "current": app.state.model_manager.current_model_id
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": app.state.model_manager.current_model_id is not None,
        "current_model": app.state.model_manager.current_model_id,
        "device": str(app.state.model_manager.device)
    }


@app.get("/api/stats")
async def get_stats():
    """Get server statistics."""
    return {
        "models_cached": len(app.state.model_manager.models),
        "active_sessions": len(app.state.sessions),
        "device": str(app.state.model_manager.device)
    }


# ====================
# Frontend Serving
# ====================

# Determine project root and frontend dist path
scripts_dir = Path(__file__).parent
project_root = scripts_dir.parent
frontend_dist_path = project_root / "prebuilt_frontend" / "dist"

logger.info(f"Frontend dist path: {frontend_dist_path}")

# Serve static assets if frontend is built and not in dev mode
if frontend_dist_path.exists() and not args.dev_mode:
    logger.info(f"Serving frontend from: {frontend_dist_path}")
    assets_path = frontend_dist_path / "assets"
    if assets_path.exists():
        app.mount("/assets", StaticFiles(directory=assets_path, html=True), name="frontend-assets")
    else:
        logger.warning(f"Assets directory not found at: {assets_path}")
else:
    logger.info("DEV MODE or frontend not built: Skipping frontend serving")


@app.get("/{path_name:path}", response_class=HTMLResponse)
async def spa_fallback(path_name: str):
    """
    Serve the frontend Single-Page Application (SPA) index.html file for all non-API routes.

    This endpoint acts as a fallback handler for client-side routing in modern web apps.
    If the requested path is not an API route, it returns the compiled `index.html` file
    from the frontend distribution directory, allowing the SPA router to handle navigation.

    Args:
        path_name: The requested path after the root (e.g., '/about', '/chat')

    Returns:
        HTMLResponse: The contents of the frontend index.html file

    Raises:
        HTTPException: If the path is an API route (404) or index.html not found (500)
    """
    # API routes should be handled by FastAPI routers, not the SPA
    if path_name.startswith("api/"):
        raise HTTPException(status_code=404, detail="API route not found")

    # In dev mode, return helpful message
    if args.dev_mode:
        return HTMLResponse("""
            <html>
                <head><title>MoE Chat Server - Dev Mode</title></head>
                <body style="font-family: system-ui; padding: 2rem; max-width: 800px; margin: 0 auto;">
                    <h1>🚀 MoE Chat Server - Development Mode</h1>
                    <p>The server is running in development mode.</p>
                    <h2>API Endpoints:</h2>
                    <ul>
                        <li><a href="/api/health">/api/health</a> - Health check</li>
                        <li><a href="/api/models">/api/models</a> - List available models</li>
                        <li><a href="/api/stats">/api/stats</a> - Server statistics</li>
                        <li><a href="/docs">/docs</a> - API documentation</li>
                    </ul>
                    <h2>Frontend:</h2>
                    <p>To start the frontend in dev mode:</p>
                    <pre style="background: #f5f5f5; padding: 1rem; border-radius: 4px;">cd frontend && npm install && npm run dev</pre>
                    <p>The frontend will be available at <code>http://localhost:5173</code></p>
                    <h2>Production Build:</h2>
                    <p>To build and serve the frontend:</p>
                    <pre style="background: #f5f5f5; padding: 1rem; border-radius: 4px;">cd frontend && npm run build
python -m scripts.chat_server</pre>
                </body>
            </html>
        """)

    # Serve index.html from prebuilt_frontend/dist/
    index_file_path = frontend_dist_path / "index.html"
    if index_file_path.exists():
        with open(index_file_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    else:
        logger.error(f"index.html not found at: {index_file_path}")
        return HTMLResponse("""
            <html>
                <head><title>Frontend Not Built</title></head>
                <body style="font-family: system-ui; padding: 2rem; max-width: 800px; margin: 0 auto;">
                    <h1>⚠️ Frontend Not Built</h1>
                    <p>The React frontend has not been built yet.</p>
                    <p>To build the frontend, run:</p>
                    <pre style="background: #f5f5f5; padding: 1rem; border-radius: 4px;">cd frontend
npm install
npm run build</pre>
                    <p>Then restart the server:</p>
                    <pre style="background: #f5f5f5; padding: 1rem; border-radius: 4px;">python -m scripts.chat_server</pre>
                    <p>API documentation is available at <a href="/docs">/docs</a></p>
                </body>
            </html>
        """, status_code=500)


# ====================
# Main
# ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        reload=True if args.dev_mode else False
    )
