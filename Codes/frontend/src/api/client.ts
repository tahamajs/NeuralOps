// src/api/client.ts
import axios from 'axios'

// Get proxy URL from environment or localStorage (for runtime toggle)
const getProxyUrl = () => {
  // Priority: localStorage (user setting) > env variable
  const stored = localStorage.getItem('proxy_url')
  if (stored) return stored
  return import.meta.env.VITE_API_PROXY_URL || ''
}

const apiBaseURL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'

const api = axios.create({
  baseURL: apiBaseURL,
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
  withCredentials: true,
})

// Request interceptor to optionally use proxy
api.interceptors.request.use((config) => {
  const proxyUrl = getProxyUrl()
  if (proxyUrl) {
    // Prepend proxy URL to the full request URL
    const originalUrl = `${config.baseURL}${config.url}`
    config.url = `${proxyUrl}${originalUrl}`
    config.baseURL = '' // clear baseURL to avoid duplication
  }
  const token = localStorage.getItem('token')
  if (token) config.headers.Authorization = `Token ${token}`
  return config
})

// Response interceptor unchanged...
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token')
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

export default api

// Helper to update proxy at runtime
export const setProxyUrl = (url: string) => {
  if (url) localStorage.setItem('proxy_url', url)
  else localStorage.removeItem('proxy_url')
}

export const getProxyUrl = () => localStorage.getItem('proxy_url') || ''