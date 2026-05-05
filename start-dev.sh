#!/bin/bash
# Development startup script for backend and frontend

set -e

echo "🚀 Starting Neural Ops Development Stack..."

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Backend setup
echo -e "${BLUE}📦 Backend Setup${NC}"
cd "$(dirname "$0")/Codes/django_api"

# Install dependencies if needed
if [ ! -d "venv" ] && [ ! -d "../.venv" ]; then
    echo "Installing backend dependencies..."
    pip install -r requirements.txt
fi

# Database migrations
echo "Running database migrations..."
python manage.py migrate --noinput 2>/dev/null || true

# Start backend in background
echo -e "${GREEN}✓ Starting Django backend on http://localhost:8000${NC}"
daphne -b 0.0.0.0 -p 8000 django_api.asgi:application &
BACKEND_PID=$!

# Frontend setup
echo -e "${BLUE}📦 Frontend Setup${NC}"
cd "../../Codes/frontend"

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
fi

# Start frontend dev server
echo -e "${GREEN}✓ Starting React frontend on http://localhost:3000${NC}"
npm run dev &
FRONTEND_PID=$!

# Trap to kill both processes on exit
trap "kill $BACKEND_PID $FRONTEND_PID" EXIT

echo -e "${GREEN}🎉 Stack Ready!${NC}"
echo "Frontend: http://localhost:3000"
echo "Backend API: http://localhost:8000/api"
echo "WebSocket: ws://localhost:8000/ws"
echo ""
echo "Press Ctrl+C to stop..."

wait
