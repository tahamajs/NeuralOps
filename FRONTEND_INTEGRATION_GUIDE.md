# Frontend Integration Guide

## Quick Start

### 1. Install Dependencies
```bash
cd Codes/frontend
npm install
```

### 2. Setup Environment Variables
```bash
# Copy the example env file
cp .env.development .env.local

# The file should contain:
# VITE_API_URL=http://localhost:8000/api
# VITE_WS_URL=ws://localhost:8000
```

### 3. Run Development Server
```bash
# Starts on http://localhost:3000
npm run dev

# With specific host/port
npm run dev -- --host 0.0.0.0 --port 3000
```

### 4. Build for Production
```bash
npm run build
# Output in dist/ directory
```

## API Integration

### Authentication
The frontend automatically handles token management:

1. **Login Flow**
   - Call `api.post('/auth/login/', { username, password })`
   - Token is automatically stored in localStorage
   - Token is automatically included in all subsequent requests

2. **Token Storage**
   - Token is stored in `localStorage.getItem('token')`
   - Automatically cleared on 401 Unauthorized
   - Can be manually cleared: `localStorage.removeItem('token')`

### Making API Calls
```typescript
import api from '@/api/client'

// GET request
const response = await api.get('/experiments/')

// POST request
const response = await api.post('/train/', {
  model_name: 'distilbert-base-uncased',
  task_name: 'mrpc',
})

// PATCH request
const response = await api.patch('/experiments/123/', {
  name: 'New name',
})

// File upload with authentication
const formData = new FormData()
formData.append('file', file)
const response = await api.post('/upload/', formData, {
  headers: { 'Content-Type': 'multipart/form-data' },
})
```

### Error Handling
```typescript
try {
  const response = await api.get('/experiments/')
} catch (error: AxiosError) {
  if (error.response?.status === 401) {
    // Unauthorized - redirect to login
    window.location.href = '/login'
  } else if (error.response?.status === 404) {
    // Not found
    console.error('Resource not found')
  } else if (error.response?.status === 500) {
    // Server error
    console.error('Server error:', error.response.data)
  }
}
```

## WebSocket Integration

### Live Experiment Updates
```typescript
import { useWebSocket } from '@/hooks/useWebSocket'
import { useLiveStore } from '@/store/experimentStore'

export function ExperimentDetail({ experimentId }: Props) {
  useWebSocket(experimentId)  // Automatically connects
  const store = useLiveStore()
  
  return (
    <div>
      <h2>Status: {store.status}</h2>
      <p>Progress: {store.progress.current}/{store.progress.total}</p>
      <div>
        {store.logs.map((log, i) => <p key={i}>{log}</p>)}
      </div>
    </div>
  )
}
```

### WebSocket Message Types
- `status` - Job status update (queued, running, completed, failed)
- `log` - Log message from training
- `metrics` - Training metrics (loss, accuracy)
- `progress` - Progress update (current_step, total_steps, eta_seconds)
- `gpu` - GPU memory update
- `final` - Final metrics when job completes

## State Management

### Zustand Store (`experimentStore.ts`)
```typescript
import { useLiveStore } from '@/store/experimentStore'

export function MyComponent() {
  const store = useLiveStore()
  
  // Update methods available:
  store.updateStatus('running')
  store.addLog('New log message')
  store.updateMetrics({ loss: 0.5, accuracy: 0.92 })
  store.updateProgress({ current_step: 100, total_steps: 1000 })
  store.updateGpu({ memory_used_gb: 4.5, memory_total_gb: 8 })
  store.reset() // Clear all
  
  // Access state:
  console.log(store.status)
  console.log(store.logs)
  console.log(store.metrics)
  console.log(store.progress)
  console.log(store.gpuInfo)
}
```

## Development Setup

### Proxy Configuration
The Vite development server automatically proxies:
- `/api/*` → `http://localhost:8000/api/`
- `/ws/*` → `ws://localhost:8000/ws/`

No manual proxy configuration needed for development!

### Hot Module Replacement (HMR)
- Automatic reload on file changes
- File saves trigger instant refresh
- Preserves component state when possible

## Component Structure

### Pages
- `DashboardPage` - System overview and recent experiments
- `ExperimentListPage` - Browse all experiments with filtering
- `ExperimentDetailPage` - Real-time experiment monitoring
- `LauncherPage` - Create and configure new jobs

### Hooks
- `useWebSocket()` - Connect to live updates
- Standard React hooks from React Query

### Store
- `useLiveStore()` - Live experiment data (Zustand)

### API
- `api` - Axios instance with auth interceptors

## Environment Variables

### Development (`.env.development`)
```env
VITE_API_URL=http://localhost:8000/api
VITE_WS_URL=ws://localhost:8000
```

### Production (`.env.production`)
```env
VITE_API_URL=https://api.your-domain.com/api
VITE_WS_URL=wss://api.your-domain.com
```

### Build-time Variables
```bash
# Build with custom API URL
VITE_API_URL=https://api.example.com npm run build
```

## Common Issues

### "Failed to connect to API"
- Ensure backend is running on `http://localhost:8000`
- Check `VITE_API_URL` in `.env.local`
- Verify CORS is enabled: `CORS_ALLOW_ALL_ORIGINS=1`

### "WebSocket connection failed"
- Ensure WebSocket path matches: `/ws/experiments/{id}/`
- Check that Daphne is running (not Django dev server)
- Verify `VITE_WS_URL` in `.env.local`

### "401 Unauthorized" errors
- Token may have expired
- Check that token is stored in localStorage
- Verify Authorization header format: `Bearer {token}`

### "Cannot find module" errors
```bash
# Clear node_modules and reinstall
rm -rf node_modules
npm install
```

### Port 3000 already in use
```bash
# Use different port
npm run dev -- --port 3001

# Or kill the process using port 3000
lsof -ti:3000 | xargs kill -9
```

## Testing API Integration

### Run Integration Tests
```typescript
import { runIntegrationTests } from '@/lib/integrationTests'

// In a component or during app initialization
await runIntegrationTests()
```

### Manual API Test with cURL
```bash
# Login
TOKEN=$(curl -s -X POST http://localhost:8000/api/auth/login/ \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin"}' \
  | jq -r '.token')

# Use token
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/experiments/
```

## Production Build

### Build Output
```bash
npm run build
```

Creates optimized bundle in `dist/` with:
- Code splitting
- CSS minification
- JavaScript minification
- Source maps

### Serve Production Build Locally
```bash
npm run preview
```

## TypeScript Support

All components are fully typed TypeScript with React 18 strict mode.

### Type Definitions
- API responses: `Experiment`, `TrainingRun`, etc.
- WebSocket messages: Typed in `useWebSocket.ts`
- Store state: Typed in `experimentStore.ts`

### Type Checking
```bash
# Check types (built into IDE)
npx tsc --noEmit
```

## Performance Optimization

### React Query (TanStack Query)
- Automatic caching of API responses
- Background refetching
- Deduplication of duplicate requests
- Configure in `main.tsx` QueryClientProvider

### WebSocket
- Automatic reconnection on disconnect
- Resource cleanup on unmount
- Efficient JSON parsing

### Code Splitting
- Page components lazy loaded
- Plugins loaded on demand
- Automatic with Vite

## Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
- Requires modern JavaScript (ES2020+)

## Deployment

### Vercel
```bash
# Connect your repo and auto-deploy
vercel
```

### Docker
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json .
RUN npm install
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "run", "preview"]
```

### Nginx
```nginx
server {
    listen 80;
    server_name your-domain.com;
    root /var/www/frontend/dist;
    
    location / {
        try_files $uri $uri/ /index.html;
    }
    
    location /api/ {
        proxy_pass http://backend:8000;
    }
    
    location /ws {
        proxy_pass http://backend:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

## Debugging

### React DevTools
- Install React DevTools browser extension
- Inspect component hierarchy
- Track state changes

### Network Tab (Browser DevTools)
- Monitor API requests/responses
- Check WebSocket frames
- Verify token headers

### Console Logs
```typescript
// API client logs auth state
console.log(localStorage.getItem('token'))

// WebSocket connection status
const ws = useWebSocket(experimentId)
console.log(ws.current?.readyState)
```

## Resources

- [Vite Documentation](https://vitejs.dev)
- [React Documentation](https://react.dev)
- [Zustand Store](https://github.com/pmndrs/zustand)
- [React Query](https://tanstack.com/query/latest)
- [Axios Documentation](https://axios-http.com)
