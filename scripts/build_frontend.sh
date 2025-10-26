#!/bin/bash
#
# Build Frontend Script
#
# Builds the React frontend and outputs to prebuilt_frontend/dist/
# for serving by the FastAPI server.
#

set -e  # Exit on error

echo "🚀 Building MoE LLaMA Frontend..."
echo ""

# Check if frontend directory exists
if [ ! -d "frontend" ]; then
    echo "❌ Error: frontend/ directory not found"
    echo "   Please run this script from the project root"
    exit 1
fi

# Navigate to frontend
cd frontend

# Check if node_modules exists, if not install dependencies
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies with yarn (first time setup)..."
    yarn install
    echo ""
fi

# Build the frontend
echo "🔨 Building React app..."
yarn build

# Check if build was successful
if [ -d "../prebuilt_frontend/dist" ]; then
    echo ""
    echo "✅ Build successful!"
    echo "   Output: prebuilt_frontend/dist/"
    echo ""
    echo "📊 Build statistics:"
    du -sh ../prebuilt_frontend/dist
    echo ""
    echo "🎉 Ready to serve!"
    echo "   Run: python -m scripts.chat_server"
else
    echo "❌ Build failed - output directory not found"
    exit 1
fi

cd ..
