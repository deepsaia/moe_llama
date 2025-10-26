## Chat UI Guide

Comprehensive guide for the MoE Chat Interface - a web-based chat UI for interacting with your trained models.

## 🎯 Overview

The chat interface provides:
- **Streaming responses** - See text generated in real-time
- **Model selection** - Switch between trained checkpoints
- **Session management** - Track conversations
- **Modern UI** - Built with React and assistant-ui
- **Single server** - Frontend and API served from one FastAPI instance

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│         Browser (React App)              │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │  Chat Interface (assistant-ui)     │ │
│  │  - Message display                 │ │
│  │  - Input box                       │ │
│  │  - Model selector                  │ │
│  └────────────────────────────────────┘ │
└──────────────┬──────────────────────────┘
               │ HTTP/SSE
               │
┌──────────────▼──────────────────────────┐
│      FastAPI Server                     │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │  API Endpoints                     │ │
│  │  - /api/chat/completions (SSE)     │ │
│  │  - /api/models (GET)               │ │
│  │  - /api/health (GET)               │ │
│  └────────────────────────────────────┘ │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │  ModelManager                      │ │
│  │  - Load models from trained_models/│ │
│  │  - Cache loaded models             │ │
│  │  - Handle model switching          │ │
│  └────────────────────────────────────┘ │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │  Static File Serving               │ │
│  │  - Serve React build (/)           │ │
│  │  - Serve assets (/assets)          │ │
│  └────────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Train a Model

First, you need a trained model:

```bash
python -m scripts.train
```

This creates checkpoints in `trained_models/`.

### 2. Start the Server

```bash
# Basic
python -m scripts.chat_server

# Custom port
python -m scripts.chat_server --port 8080

# Bind to all interfaces (for remote access)
python -m scripts.chat_server --host 0.0.0.0
```

### 3. Open the UI

Navigate to:
```
http://localhost:8000
```

## 📁 Project Structure

```
moe_llama/
├── scripts/
│   └── chat_server.py       # FastAPI server with streaming API
├── frontend/                 # React application
│   ├── src/
│   │   ├── App.tsx          # Main application
│   │   ├── components/      # UI components
│   │   └── api/             # API client
│   ├── package.json
│   └── dist/                # Built frontend (served by FastAPI)
├── trained_models/          # Your model checkpoints
│   ├── model_tiny_shakespeare_20250126.pt
│   └── vocab_tiny_shakespeare.txt
└── docs/
    └── chat_ui.md           # This file
```

## 🎨 Frontend Setup

### Why assistant-ui?

We chose [assistant-ui](https://github.com/assistant-ui/assistant-ui) because:
- **Modern**: Built for React with TypeScript
- **Flexible**: Works with any backend
- **Streaming**: First-class SSE support
- **Customizable**: Easy to style and extend
- **Well-maintained**: Active development and community

### Initial Setup

```bash
cd frontend

# Install dependencies
npm install

# Development (with hot reload)
npm run dev

# Production build
npm run build
```

### Development vs Production

**Development Mode:**
```bash
# Terminal 1: Run FastAPI server
python -m scripts.chat_server --dev-mode

# Terminal 2: Run React dev server
cd frontend && npm run dev
# Opens at http://localhost:5173
```

**Production Mode:**
```bash
# Build frontend once
cd frontend && npm run build

# Run server (serves built React app)
python -m scripts.chat_server
# Opens at http://localhost:8000
```

## 🔌 API Endpoints

### POST /api/chat/completions

Streaming chat completions endpoint.

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "How are you?"}
  ],
  "model": "model_tiny_shakespeare_20250126",
  "temperature": 0.8,
  "max_tokens": 200,
  "top_k": 50,
  "top_p": 0.95,
  "session_id": "optional-session-id"
}
```

**Response (Server-Sent Events):**
```
data: {"content": "I'm"}

data: {"content": " doing"}

data: {"content": " well"}

data: {"done": true}
```

**Parameters:**
- `messages` (required): Conversation history
- `model` (optional): Model ID, defaults to most recent
- `temperature` (optional): 0.0-2.0, controls randomness
- `max_tokens` (optional): 1-2048, maximum tokens to generate
- `top_k` (optional): 1-200, top-k sampling
- `top_p` (optional): 0.0-1.0, nucleus sampling
- `session_id` (optional): For tracking conversations

### GET /api/models

List available models.

**Response:**
```json
{
  "models": [
    {
      "id": "model_tiny_shakespeare_20250126",
      "name": "model_tiny_shakespeare_20250126.pt",
      "size": 12345678,
      "modified": "2025-01-26T10:30:00"
    }
  ],
  "current": "model_tiny_shakespeare_20250126"
}
```

### GET /api/health

Health check.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "current_model": "model_tiny_shakespeare_20250126",
  "device": "cuda:0"
}
```

### GET /api/stats

Server statistics.

**Response:**
```json
{
  "models_cached": 2,
  "active_sessions": 5,
  "device": "cuda:0"
}
```

## 🛠️ Frontend Implementation

### Key Files

**`frontend/src/App.tsx`** - Main application component
```typescript
import { AssistantRuntimeProvider } from "@assistant-ui/react";

export function App() {
  return (
    <AssistantRuntimeProvider runtime={...}>
      <ChatInterface />
    </AssistantRuntimeProvider>
  );
}
```

**`frontend/src/api/client.ts`** - API client for backend
```typescript
export async function* streamChat(messages, options) {
  const response = await fetch('/api/chat/completions', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({messages, ...options})
  });

  const reader = response.body.getReader();
  // Parse SSE stream...
}
```

**`frontend/src/components/ModelSelector.tsx`** - Model dropdown
```typescript
export function ModelSelector() {
  const [models, setModels] = useState([]);

  useEffect(() => {
    fetch('/api/models')
      .then(r => r.json())
      .then(data => setModels(data.models));
  }, []);

  return <select>{/* model options */}</select>;
}
```

## 🎨 Customization

### Styling

The UI uses Tailwind CSS. Customize in `frontend/tailwind.config.js`:

```javascript
module.exports = {
  theme: {
    extend: {
      colors: {
        primary: '#your-color',
      }
    }
  }
}
```

### Adding Features

**User Authentication:**
```typescript
// Add auth header to API calls
headers: {
  'Authorization': `Bearer ${token}`
}
```

**Message History:**
```typescript
// Save to localStorage
localStorage.setItem('chat-history', JSON.stringify(messages));
```

**Custom System Prompts:**
```typescript
const systemPrompt = "You are a helpful assistant...";
messages.unshift({role: 'system', content: systemPrompt});
```

## 📊 Model Management

### Adding Models

Models are automatically discovered from `trained_models/`:

```
trained_models/
├── model_dataset1_20250126_120000.pt
├── vocab_dataset1.txt
├── model_dataset2_20250126_140000.pt
└── vocab_dataset2.txt
```

Naming convention:
- Models: `model_{dataset}_{timestamp}.pt`
- Vocab: `vocab_{dataset}.txt`

### Switching Models

**Via UI:** Use the model selector dropdown

**Via API:**
```json
{
  "messages": [...],
  "model": "model_tiny_shakespeare_20250126"
}
```

### Model Caching

Loaded models are cached in memory. To clear cache, restart the server.

## 🔒 Security & Limits

### Request Limits

To prevent abuse:
- Max messages per request: 100
- Max message length: 4,000 characters
- Max total conversation: 16,000 characters
- Temperature: 0.0 - 2.0
- Top-k: 1 - 200
- Max tokens: 1 - 2,048

### Production Deployment

For production use:

```bash
# Use reverse proxy (nginx)
server {
  listen 80;
  location / {
    proxy_pass http://localhost:8000;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
  }
}

# Add rate limiting
# Add authentication
# Use HTTPS
```

## 🐛 Troubleshooting

### Server won't start

**Issue:** `No trained models found`

**Solution:**
```bash
python -m scripts.train  # Train a model first
```

### Frontend not loading

**Issue:** Blank page at http://localhost:8000

**Solution:**
```bash
cd frontend
npm run build
# Then restart server
```

### Streaming not working

**Issue:** Response doesn't stream, appears all at once

**Solution:** Check browser dev tools. SSE requires proper headers:
```
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
```

### Model switching fails

**Issue:** Error when selecting different model

**Solution:** Ensure vocab file exists for the model:
```
trained_models/
├── model_dataset_20250126.pt
└── vocab_dataset.txt  # Must match dataset name
```

## 📚 Examples

### Basic Chat

```javascript
// Send message
const response = await fetch('/api/chat/completions', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    messages: [
      {role: 'user', content: 'Hello!'}
    ]
  })
});

// Read stream
const reader = response.body.getReader();
while (true) {
  const {done, value} = await reader.read();
  if (done) break;
  // Process chunk...
}
```

### Custom Generation

```javascript
// Higher temperature for creativity
{
  messages: [...],
  temperature: 1.2,
  max_tokens: 500,
  top_p: 0.9
}

// Lower temperature for consistency
{
  messages: [...],
  temperature: 0.3,
  max_tokens: 100,
  top_k: 10
}
```

### Session Tracking

```javascript
const sessionId = crypto.randomUUID();

// All requests with same sessionId
{
  messages: [...],
  session_id: sessionId
}
```

## 🚀 Advanced

### Multi-GPU Support

```python
# chat_server.py - Add worker pool like nanochat
class WorkerPool:
    def __init__(self, num_gpus):
        self.workers = []
        for gpu_id in range(num_gpus):
            device = torch.device(f"cuda:{gpu_id}")
            model, tokenizer = load_model(device)
            self.workers.append((model, tokenizer))
```

### Custom UI Components

```typescript
// frontend/src/components/CustomMessage.tsx
export function CustomMessage({message}) {
  return (
    <div className="message">
      <Avatar role={message.role} />
      <MessageContent>{message.content}</MessageContent>
      <Timestamp>{message.timestamp}</Timestamp>
    </div>
  );
}
```

### Analytics

```python
# Add to chat_server.py
@app.middleware("http")
async def log_requests(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    logger.info(f"{request.method} {request.url} - {duration:.2f}s")
    return response
```

## 📖 References

- [assistant-ui Documentation](https://github.com/assistant-ui/assistant-ui)
- [FastAPI Streaming](https://fastapi.tiangolo.com/advanced/custom-response/)
- [Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)
- [React TypeScript](https://react.dev/learn/typescript)

## 🤝 Contributing

To contribute UI improvements:

1. Fork the repository
2. Create feature branch
3. Implement changes in `frontend/src`
4. Test with `npm run dev`
5. Build with `npm run build`
6. Submit pull request

## ✨ Future Enhancements

Planned features:
- **Authentication** - User login and API keys
- **Chat history** - Persistent conversations
- **Multi-modal** - Image support
- **Voice input** - Speech-to-text
- **Export** - Download conversations
- **Themes** - Dark mode, custom themes
- **Plugins** - Extensibility system

---

**Happy Chatting! 💬**
