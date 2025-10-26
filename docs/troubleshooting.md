# Troubleshooting Guide

## Current Issues & Fixes

### 1. Vocab File Not Found

**Fixed**: Updated `scripts/chat_server.py` to properly handle model files with timestamps in format `model_{dataset}_{YYYYMMDD}_{HHMMSS}.pt`

**How it works now:**
- Model: `model_tiny_shakespeare_20251026_070939.pt`
- Extracts: `tiny_shakespeare` (stops at first numeric part >= 6 digits)
- Looks for: `vocab_tiny_shakespeare.txt` ✅

**To test:**
```bash
# Restart server
python -m scripts.chat_server

# Should see:
# ✅ Loading model: trained_models/model_tiny_shakespeare_20251026_070939.pt
# ✅ Loading vocab: trained_models/vocab_tiny_shakespeare.txt
```

### 2. Cramped Chat UI

**Need to debug** - Please provide:

1. Open browser dev tools (F12)
2. Navigate to Elements/Inspector tab
3. Find the element with class `assistant-thread`
4. Check its computed height

OR take a screenshot showing the HTML structure.

### 3. Steps to Restart Everything

```bash
# 1. Stop the backend server (Ctrl+C)

# 2. If in dev mode, stop yarn dev (Ctrl+C)

# 3. Rebuild frontend
cd frontend
yarn build
cd ..

# 4. Restart backend
python -m scripts.chat_server

# 5. Hard refresh browser
# Mac: Cmd+Shift+R
# Windows/Linux: Ctrl+Shift+R

# 6. Check browser console for errors
```

### 4. Check Server Logs

When you start the server, you should see:
```
INFO - Starting MoE Chat Server...
INFO - ModelManager initialized. Device: cpu
INFO - Model directory: trained_models
INFO - Loading default (most recent) model: model_tiny_shakespeare_20251026_070939
INFO - Loading model: trained_models/model_tiny_shakespeare_20251026_070939.pt
INFO - Loading vocab: trained_models/vocab_tiny_shakespeare.txt
INFO - Model loaded successfully: model_tiny_shakespeare_20251026_070939
INFO - Server ready at http://127.0.0.1:8000
```

### 5. Test API Endpoints

```bash
# Test health
curl http://localhost:8000/api/health

# Test models list
curl http://localhost:8000/api/models

# Test chat (requires model loaded)
curl -X POST http://localhost:8000/api/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}]}'
```

### 6. Common Issues

**404 on /api/chat/completions:**
- Make sure server started successfully
- Check model loaded without errors
- Verify no other service on port 8000

**Vocab file not found:**
- Fixed in latest code
- Restart server to apply fix

**UI cramped:**
- Hard refresh browser (Cmd+Shift+R)
- Check browser console for CSS errors
- Verify prebuilt_frontend/dist exists
