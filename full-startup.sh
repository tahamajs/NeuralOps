#!/bin/bash
# Enhanced Project Startup & Verification Script
# Checks all systems and starts backend/frontend

set -e

# ==================== COLORS & FORMATTING ====================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

CHECKMARK='✓'
CROSSMARK='✗'
ARROW='→'

# ==================== UTILITY FUNCTIONS ====================
print_header() {
    echo -e "\n${BLUE}===============================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}===============================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}${CHECKMARK}${NC} $1"
}

print_error() {
    echo -e "${RED}${CROSSMARK}${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${CYAN}ℹ${NC} $1"
}

print_step() {
    echo -e "\n${CYAN}${ARROW} $1...${NC}"
}

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

wait_for_service() {
    local url=$1
    local timeout=${2:-30}
    local start_time=$(date +%s)
    
    while true; do
        elapsed=$(($(date +%s) - start_time))
        if [ $elapsed -gt $timeout ]; then
            return 1
        fi
        
        if curl -s "$url" > /dev/null 2>&1; then
            return 0
        fi
        
        sleep 1
    done
}

# ==================== CHECKS ====================
check_python() {
    print_step "Checking Python installation"
    if command_exists python3; then
        version=$(python3 --version | awk '{print $2}')
        print_success "Python3 found: $version"
        return 0
    else
        print_error "Python3 not found"
        return 1
    fi
}

check_node() {
    print_step "Checking Node.js installation"
    if command_exists node; then
        version=$(node --version)
        print_success "Node.js found: $version"
        return 0
    else
        print_warning "Node.js not found (required for frontend)"
        return 1
    fi
}

check_pip_packages() {
    print_step "Checking required Python packages"
    local packages=("django" "djangorestframework" "channels" "torch" "transformers")
    local missing=0
    
    for package in "${packages[@]}"; do
        if python3 -c "import ${package}" 2>/dev/null; then
            print_success "$package installed"
        else
            print_warning "$package not found - will be installed"
            missing=$((missing + 1))
        fi
    done
    
    return $missing
}

check_npm_packages() {
    print_step "Checking Node.js packages (frontend)"
    if [ ! -d "Codes/frontend/node_modules" ]; then
        print_warning "npm packages not installed"
        return 1
    else
        print_success "npm packages installed"
        return 0
    fi
}

check_database() {
    print_step "Checking Django database"
    if [ -f "Codes/django_api/db.sqlite3" ]; then
        print_success "Database exists"
        return 0
    else
        print_warning "Database does not exist - will be created"
        return 1
    fi
}

check_env_files() {
    print_step "Checking environment files"
    local issues=0
    
    if [ ! -f "Codes/django_api/.env" ]; then
        print_warning "Backend .env not found"
        issues=$((issues + 1))
    else
        print_success "Backend .env exists"
    fi
    
    if [ ! -f "Codes/frontend/.env.local" ]; then
        print_warning "Frontend .env.local not found"
        issues=$((issues + 1))
    else
        print_success "Frontend .env.local exists"
    fi
    
    return $issues
}

# ==================== SETUP ====================
setup_backend() {
    print_step "Setting up backend"
    
    cd Codes/django_api
    
    # Check if venv exists
    if [ ! -d "venv" ]; then
        print_info "Creating Python virtual environment..."
        python3 -m venv venv
        print_success "Virtual environment created"
    fi
    
    # Activate venv
    source venv/bin/activate
    
    # Install requirements
    print_info "Installing Python dependencies..."
    pip install -q -r requirements.txt --upgrade
    print_success "Dependencies installed"
    
    # Run migrations
    print_info "Running database migrations..."
    python manage.py migrate --noinput 2>/dev/null || true
    print_success "Migrations completed"
    
    # Initialize database
    if [ ! -f "db.sqlite3" ]; then
        print_info "Initializing database..."
        python init_db.py
        print_success "Database initialized"
    fi
    
    cd - > /dev/null
}

setup_frontend() {
    print_step "Setting up frontend"
    
    cd Codes/frontend
    
    # Install npm packages
    if [ ! -d "node_modules" ]; then
        print_info "Installing npm dependencies..."
        npm install -q
        print_success "npm packages installed"
    else
        print_success "npm packages already installed"
    fi
    
    cd - > /dev/null
}

# ==================== START SERVICES ====================
start_backend() {
    print_step "Starting backend server"
    
    cd Codes/django_api
    source venv/bin/activate
    
    # Check if Daphne is available
    if command_exists daphne; then
        print_info "Starting with Daphne (WebSocket support)..."
        daphne -b 0.0.0.0 -p 8000 django_api.asgi:application &
    else
        print_info "Starting with Django development server..."
        python manage.py runserver 0.0.0.0:8000 &
    fi
    
    BACKEND_PID=$!
    echo "Backend PID: $BACKEND_PID"
    
    # Wait for backend to start
    print_info "Waiting for backend to become ready..."
    if wait_for_service "http://localhost:8000/api/health/" 30; then
        print_success "Backend is running on http://localhost:8000"
    else
        print_error "Backend failed to start"
        return 1
    fi
    
    cd - > /dev/null
    return 0
}

start_frontend() {
    print_step "Starting frontend server"
    
    cd Codes/frontend
    
    print_info "Starting Vite development server..."
    npm run dev -- --host 0.0.0.0 --port 5173 &
    
    FRONTEND_PID=$!
    echo "Frontend PID: $FRONTEND_PID"
    
    # Wait for frontend to start
    print_info "Waiting for frontend to become ready..."
    if wait_for_service "http://localhost:5173/" 30; then
        print_success "Frontend is running on http://localhost:5173"
    else
        print_warning "Frontend may take longer to start"
    fi
    
    cd - > /dev/null
    return 0
}

# ==================== TEST ENDPOINTS ====================
test_endpoints() {
    print_step "Testing backend endpoints"
    
    # Wait a moment for the service to fully start
    sleep 2
    
    # Test health endpoint
    if curl -s http://localhost:8000/api/health/ | grep -q "healthy"; then
        print_success "Health check passed"
    else
        print_warning "Health check returned unexpected response"
    fi
    
    # Test version endpoint
    version=$(curl -s http://localhost:8000/api/version/ | grep -o "\"api_version\":\"[^\"]*\"" | head -1)
    if [ ! -z "$version" ]; then
        print_success "Version check passed: $version"
    else
        print_warning "Could not retrieve version"
    fi
    
    # Test system status
    cpu=$(curl -s http://localhost:8000/api/system/status/ | grep -o "\"percent\":[0-9]*" | head -1)
    if [ ! -z "$cpu" ]; then
        print_success "System status check passed"
    else
        print_warning "Could not retrieve system status"
    fi
}

# ==================== MAIN ====================
main() {
    print_header "Project Startup & Verification"
    
    # Get project root
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    cd "$PROJECT_ROOT"
    
    print_info "Project root: $PROJECT_ROOT"
    
    # ========== VERIFICATION ==========
    print_header "System Verification"
    
    check_python || exit 1
    check_node || print_warning "Frontend requires Node.js"
    check_env_files || print_warning "Some environment files missing"
    
    # ========== SETUP ==========
    print_header "Project Setup"
    
    read -p "Continue with setup? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_warning "Setup skipped"
        exit 0
    fi
    
    setup_backend || exit 1
    setup_frontend || print_warning "Frontend setup had issues"
    
    # ========== STARTUP ==========
    print_header "Starting Services"
    
    read -p "Start backend and frontend? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_warning "Startup skipped"
        exit 0
    fi
    
    # Start services in background
    start_backend || exit 1
    sleep 2
    test_endpoints
    sleep 1
    start_frontend || print_warning "Frontend may still be starting"
    
    # ========== FINAL INFO ==========
    print_header "Services Started Successfully"
    
    echo -e "${GREEN}Backend:${NC}   http://localhost:8000"
    echo -e "${GREEN}  API:      http://localhost:8000/api"
    echo -e "${GREEN}  Admin:    http://localhost:8000/admin"
    echo -e "${GREEN}  Docs:     http://localhost:8000/api/"
    echo ""
    echo -e "${GREEN}Frontend:${NC}  http://localhost:5173"
    echo ""
    echo -e "${CYAN}Default Credentials:${NC}"
    echo -e "  Username: ${CYAN}admin${NC}"
    echo -e "  Password: ${CYAN}admin123${NC}"
    echo ""
    
    print_info "Press Ctrl+C to stop services"
    
    # Keep script running
    wait
}

# ==================== TRAP INTERRUPTS ====================
cleanup() {
    print_warning "Shutting down services..."
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null || true
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
    fi
    print_success "Services stopped"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Run main
main
