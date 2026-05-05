#!/bin/bash

# Integration Verification Script
# This script checks if backend and frontend are properly configured

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Backend & Frontend Integration Verification ===${NC}\n"

ISSUES=0

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check file exists
file_exists() {
    [ -f "$1" ]
}

# Function to print result
check_result() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $2"
    else
        echo -e "${RED}✗${NC} $2"
        ISSUES=$((ISSUES + 1))
    fi
}

# Check Python/Backend
echo -e "${BLUE}Backend Checks:${NC}"

if command_exists python3; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    check_result 0 "Python installed: v$PYTHON_VERSION"
else
    check_result 1 "Python not found"
fi

file_exists "Codes/django_api/requirements.txt"
check_result $? "Django requirements.txt exists"

file_exists "Codes/django_api/db.sqlite3" || file_exists "Codes/django_api/db.sqlite3"
if [ -f "Codes/django_api/db.sqlite3" ]; then
    check_result 0 "Database exists"
else
    echo -e "${YELLOW}⚠${NC} Database not initialized (will be created on first run)"
fi

# Check Django settings
if grep -q "INSTALLED_APPS" "Codes/django_api/django_api/settings.py"; then
    check_result 0 "Django settings configured"
else
    check_result 1 "Django settings malformed"
fi

# Check ASGI for WebSocket
if grep -q "ProtocolTypeRouter" "Codes/django_api/django_api/asgi.py"; then
    check_result 0 "ASGI WebSocket support configured"
else
    check_result 1 "ASGI WebSocket not configured"
fi

# Check API authentication
if grep -q "CookieTokenAuthentication" "Codes/django_api/api/authentication.py"; then
    check_result 0 "Token authentication configured"
else
    check_result 1 "Token authentication not configured"
fi

# Check compress endpoint
if grep -q "start_compress" "Codes/django_api/api/views.py"; then
    check_result 0 "Compress endpoint implemented"
else
    check_result 1 "Compress endpoint missing"
fi

echo ""
echo -e "${BLUE}Frontend Checks:${NC}"

# Check Node.js
if command_exists node; then
    NODE_VERSION=$(node --version)
    check_result 0 "Node.js installed: $NODE_VERSION"
else
    check_result 1 "Node.js not found"
fi

if command_exists npm; then
    NPM_VERSION=$(npm --version)
    check_result 0 "npm installed: v$NPM_VERSION"
else
    check_result 1 "npm not found"
fi

# Check package.json
file_exists "Codes/frontend/package.json"
check_result $? "package.json exists"

# Check node_modules
if [ -d "Codes/frontend/node_modules" ]; then
    check_result 0 "Dependencies installed (node_modules exists)"
else
    echo -e "${YELLOW}⚠${NC} Dependencies not installed (run: npm install)"
fi

# Check Vite config
if grep -q "proxy:" "Codes/frontend/vite.config.ts"; then
    check_result 0 "Vite proxy configured"
else
    check_result 1 "Vite proxy not configured"
fi

# Check API client
if grep -q "axios.create" "Codes/frontend/src/api/client.ts"; then
    check_result 0 "Axios client configured"
else
    check_result 1 "Axios client not configured"
fi

# Check WebSocket hook
if grep -q "useWebSocket" "Codes/frontend/src/hooks/useWebSocket.ts"; then
    check_result 0 "WebSocket hook implemented"
else
    check_result 1 "WebSocket hook not implemented"
fi

# Check environment config
file_exists "Codes/frontend/.env.development"
check_result $? "Environment config exists"

echo ""
echo -e "${BLUE}Configuration Checks:${NC}"

# Check if both are configured properly
BACKEND_READY=1
FRONTEND_READY=1

if grep -q "CORS_ALLOW_ALL_ORIGINS" "Codes/django_api/django_api/settings.py"; then
    check_result 0 "CORS enabled for development"
else
    check_result 1 "CORS not configured"
    BACKEND_READY=0
fi

if grep -q "CHANNEL_LAYERS" "Codes/django_api/django_api/settings.py"; then
    check_result 0 "Django Channels configured"
else
    check_result 1 "Django Channels not configured"
    BACKEND_READY=0
fi

if grep -q "VITE_API_URL" "Codes/frontend/.env.development"; then
    check_result 0 "Frontend API URL configured"
else
    check_result 1 "Frontend API URL not configured"
    FRONTEND_READY=0
fi

echo ""
echo -e "${BLUE}Summary:${NC}"

if [ $ISSUES -eq 0 ]; then
    echo -e "${GREEN}✅ All checks passed!${NC}\n"
    echo "You're ready to start developing!"
    echo ""
    echo "Backend startup:"
    echo "  cd Codes/django_api"
    echo "  daphne -b 0.0.0.0 -p 8000 django_api.asgi:application"
    echo ""
    echo "Frontend startup (new terminal):"
    echo "  cd Codes/frontend"
    echo "  npm run dev"
    echo ""
    echo "Then visit: http://localhost:3000"
else
    echo -e "${YELLOW}⚠⚠⚠ $ISSUES issue(s) found ⚠⚠⚠${NC}\n"
    echo "Please fix the above issues before starting development."
fi

exit $ISSUES
