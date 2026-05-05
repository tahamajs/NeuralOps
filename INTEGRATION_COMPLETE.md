# Full Backend & Frontend Integration Guide

## 🎯 Overview

Your project now has complete backend-frontend integration with:
- ✅ Full REST API with Django + DRF
- ✅ WebSocket support for real-time updates
- ✅ Token-based authentication
- ✅ React 18 + TypeScript frontend
- ✅ Automatic API client configuration
- ✅ Hot module reloading for development

---

## 🚀 Quick Setup (5 minutes)

### Backend
```bash
cd Codes/django_api
pip install -r requirements.txt
python manage.py migrate
daphne -b 0.0.0.0 -p 8000 django_api.asgi:application
```

### Frontend (new terminal)
```bash
cd Codes/frontend
npm install
npm run dev
```

That's it! 
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000/api
- WebSocket: ws://localhost:8000/ws

---

## 📝 Key Fixes Applied

### Backend
1. **Fixed Register Endpoint** - Changed `User.objects_create_user` → `User.objects.create_user`
2. **Added Compress Endpoint** - `/api/compress/` now available
3. **Fixed WebSocket Routing** - Both `/ws/experiments/{id}/` and `/ws/jobs/{id}/` work
4. **Improved Authentication** - Better token validation with proper error handling
5. **WebSocket Message Handling** - Proper JSON encoding for all message types

### Frontend  
1. **API Client Interceptors** - Auto token management and 401 handling
2. **WebSocket Integration** - Robust collection with better error handling
3. **Environment Configuration** - Flexible URL configuration via .env
4. **Development Proxy** - Vite proxy for seamless API calls
5. **Token Storage** - localStorage persistence with automatic cleanup

---

## 🔐 Authentication Flow

### 1. Register/Login
```typescript
// Frontend
const response = await api.post('/auth/login/', {
  username: 'user',
  password: 'password'
})
// Token automatically saved to localStorage
```

### 2. All Subsequent Requests
```typescript
// Token automatically included
const experiments = await api.get('/experiments/')
// Header: Authorization: Bearer {token}
```

### 3. Token Refresh
```typescript
// On 401, user is redirected to login
// Or manually refresh:
const response = await api.post('/auth/refresh-token/')
// New token automatically applied
```

---

## 📡 WebSocket Connection

### Real-time Experiment Updates
```typescript
import { useWebSocket } from '@/hooks/useWebSocket'
import { useLiveStore } from '@/store/experimentStore'

export function ExperimentDetail({ experimentId }: Props) {
  // Automatically connects and updates state
  useWebSocket(experimentId)
  const store = useLiveStore()
  
  return <div>Status: {store.status}</div>
}
```

### Message Types Supported
- `status` - Job state (queued, running, completed, failed)
- `log` - Log messages
- `metrics` - Training metrics (loss, accuracy)
- `progress` - Step progress and ETA
- `gpu` - GPU memory usage
- `final` - Final results

---

## 🗄️ Database Models

### Experiment
- `id` (UUID) - Primary key
- `name` - Experiment name
- `description` - Optional description
- `config` - JSON configuration
- `created_at`, `updated_at` - Timestamps
- `runs` - Related training runs

### TrainingRun
- `id` (UUID)
- `experiment` (FK) - Parent experiment
- `job_type` - training, evaluation, or compression
- `status` - queued, running, completed, failed, stopped
- `config` - Job configuration
- `final_metrics` - Results
- `logs` - Text log output
- `current_step` / `total_steps` - Progress
- CPU/GPU usage metrics

---

## 🔗 API Endpoints

### Auth (`/api/auth/`)
- `POST /register/` → User + Token
- `POST /login/` → Token
- `POST /logout/` → Status
- `GET /user/` → User info
- `PATCH /profile/` → Update profile
- `POST /change-password/` → Refresh token

### Config (`/api/config/`)
- `GET /models/` → Available models
- `GET /tasks/` → Available tasks
- `GET /methods/` → Compression methods
- `GET /attentions/` → Attention types
- `GET /presets/` → Training presets

### System (`/api/system/`)
- `GET /status/` → CPU/Memory/Disk info
- `GET /gpu/` → GPU information
- `GET /health/` → Health check
- `GET /version/` → API version

### Experiments (`/api/experiments/`)
- `GET /` → List all
- `POST /` → Create new
- `GET /{id}/` → Details
- `PATCH /{id}/` → Update
- `DELETE /{id}/` → Delete

### Jobs (`/api/`)
- `POST /train/` → Start training
- `POST /evaluate/` → Start evaluation
- `POST /compress/` → Start compression
- `GET /runs/{id}/` → Job details

---

## 🔧 Configuration

### Backend (.env)
```bash
DJANGO_DEBUG=1
DJANGO_SECRET_KEY=your-secret
ALLOWED_HOSTS=localhost,127.0.0.1
CORS_ALLOW_ALL_ORIGINS=1
SCHEDULER_TOTAL_GPUS=1
```

### Frontend (.env.development)
```bash
VITE_API_URL=http://localhost:8000/api
VITE_WS_URL=ws://localhost:8000
```

---

## 📊 Data Flow Example

### Launch Experiment
1. User fills form in LauncherPage
2. Clicks "Launch"
3. Frontend → `POST /api/train/` with config
4. Backend creates Experiment + TrainingRun
5. Backend starts training thread
6. Returns TrainingRun ID

### Monitor Experiment
1. Frontend navigates to ExperimentDetailPage
2. Calls `GET /api/experiments/{id}/` for initial state
3. Opens WebSocket `ws://localhost:8000/ws/experiments/{id}/`
4. Backend sends updates:
   - `status` → "running"
   - `progress` → Step 100/1000
   - `metrics` → Loss: 0.5, Accuracy: 0.92
   - `log` → Training log messages
5. Frontend updates in real-time

---

## ✅ Integration Checklist

- [x] Register view typo fixed
- [x] Compress endpoint added
- [x] WebSocket routing supports both paths
- [x] Token authentication middleware
- [x] Frontend API client with interceptors
- [x] WebSocket real-time updates
- [x] Environment configuration
- [x] Development proxy setup
- [x] CORS enabled for development

---

## 🐛 Troubleshooting

### "Cannot connect to API"
```bash
# Check backend is running
curl http://localhost:8000/api/health/

# Check CORS
# Backend: CORS_ALLOW_ALL_ORIGINS=1
```

### "Token not working"
```bash
# Check token is stored
localStorage.getItem('token')

# Check auth header sent
# DevTools → Network → Select request → Headers
# Should see: Authorization: Bearer {token}
```

### "WebSocket fails to connect"
```bash
# Must use Daphne (not Django dev server)
daphne -b 0.0.0.0 -p 8000 django_api.asgi:application

# Check WebSocket URL matches
ws://localhost:8000/ws/experiments/{experiment-id}/
```

### "Port already in use"
```bash
# Kill process on port 3000 (frontend)
lsof -ti:3000 | xargs kill -9

# Kill process on port 8000 (backend)
lsof -ti:8000 | xargs kill -9
```

---

## 📚 File Structure

### Backend
```
Codes/django_api/
├── api/
│   ├── views.py (REST endpoints)
│   ├── urls.py (API routes)
│   ├── models.py (Experiment, TrainingRun)
│   ├── serializers.py (Model serializers)
│   ├── consumers.py (WebSocket handlers)
│   ├── runners.py (Job execution)
│   └── authentication.py (Token auth)
├── django_api/
│   ├── settings.py (Django config)
│   ├── urls.py (Main routes)
│   ├── asgi.py (WebSocket config) ✅ FIXED
│   └── wsgi.py (WSGI application)
└── manage.py
```

### Frontend
```
Codes/frontend/
├── src/
│   ├── api/
│   │   └── client.ts (Axios setup) ✅ FIXED
│   ├── hooks/
│   │   └── useWebSocket.ts (WebSocket) ✅ FIXED
│   ├── pages/
│   │   ├── LauncherPage.tsx
│   │   ├── ExperimentListPage.tsx
│   │   └── ExperimentDetailPage.tsx
│   ├── components/
│   ├── store/
│   │   └── experimentStore.ts (Zustand)
│   ├── App.tsx (Router)
│   └── main.tsx (React init)
├── vite.config.ts ✅ FIXED
├── .env.development ✅ NEW
├── package.json
└── tsconfig.json
```

---

## 🎓 Next Steps

1. **Test the integration**: Use the integration tests in frontend
2. **Create login page**: Add auth UI if needed
3. **Add more endpoints**: Follow the same pattern
4. **Deploy**: See deployment guides in backend/frontend integration documents
5. **Monitor**: Use Django admin panel to view jobs

---

## 📞 Support Files

- [BACKEND_INTEGRATION_GUIDE.md](./BACKEND_INTEGRATION_GUIDE.md) - Detailed backend docs
- [FRONTEND_INTEGRATION_GUIDE.md](./FRONTEND_INTEGRATION_GUIDE.md) - Detailed frontend docs  
- [start-dev.sh](./start-dev.sh) - One-command startup script

---

## 🎉 You're All Set!

Your backend and frontend are now fully integrated!

**Start developing:**
```bash
# Terminal 1: Backend
cd Codes/django_api
daphne -b 0.0.0.0 -p 8000 django_api.asgi:application

# Terminal 2: Frontend
cd Codes/frontend
npm run dev
```

**Visit:**
- Frontend: http://localhost:3000
- Backend: http://localhost:8000
- API Docs: http://localhost:8000/api/

Happy coding! 🚀
