#!/bin/bash
# NeuralOps Full Stack Starter

echo "Starting NeuralOps Platform (Fully Synced)..."

# 1. Start Django Backend in the background
cd Codes/django_api
python manage.py runserver 0.0.0.0:8000 &
BACKEND_PID=$!
echo "Backend running on port 8000 (PID: $BACKEND_PID)"

# 2. Start React Frontend
cd ../frontend
echo "Starting Professional UI (Glassmorphism + Live API Hooks)..."
npm run dev

# Cleanup on exit
trap "kill $BACKEND_PID" EXIT
