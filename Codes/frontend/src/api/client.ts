// src/api/client.ts
import axios, { AxiosError, AxiosResponse } from 'axios'

const apiBaseURL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'

const api = axios.create({
  baseURL: apiBaseURL,
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
  withCredentials: true,
})

// ── Request interceptor: attach DRF Token ───────────────────────
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token')
    if (token) {
      // Django REST Framework expects "Token abc123", NOT "Bearer abc123"
      config.headers.Authorization = `Token ${token}`
    }
    return config
  },
  (error) => Promise.reject(error)
)

// ── Response interceptor: store token + handle 401 ──────────────
api.interceptors.response.use(
  (response: AxiosResponse) => {
    // Login / register endpoints return { token: "..." }
    if (response.data?.token) {
      localStorage.setItem('token', response.data.token)
    }
    return response
  },
  (error: AxiosError) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token')
      if (typeof window !== 'undefined') {
        window.location.href = '/login'
      }
    }
    return Promise.reject(error)
  }
)

export default api
export type { AxiosError, AxiosResponse }