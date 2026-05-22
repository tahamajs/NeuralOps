// src/api/client.ts
import axios, {
  AxiosInstance,
  AxiosRequestConfig,
  AxiosResponse,
  AxiosError,
  InternalAxiosRequestConfig,
} from 'axios'

// ----------------------------------------------------------------------
// Types
// ----------------------------------------------------------------------
interface ApiErrorResponse {
  detail?: string
  message?: string
  error?: string
}

// ----------------------------------------------------------------------
// Proxy helpers (runtime toggle)
// ----------------------------------------------------------------------
const getProxyUrl = (): string => {
  const stored = localStorage.getItem('proxy_url')
  if (stored) return stored
  return import.meta.env.VITE_API_PROXY_URL || ''
}

export const setProxyUrl = (url: string): void => {
  if (url) localStorage.setItem('proxy_url', url)
  else localStorage.removeItem('proxy_url')
}

export const getProxyUrl = (): string => localStorage.getItem('proxy_url') || ''

// ----------------------------------------------------------------------
// API base URL
// ----------------------------------------------------------------------
const apiBaseURL: string = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'

// ----------------------------------------------------------------------
// Create axios instance
// ----------------------------------------------------------------------
const api: AxiosInstance = axios.create({
  baseURL: apiBaseURL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
  withCredentials: true,
})

// ----------------------------------------------------------------------
// Request interceptor: attach token and proxy
// ----------------------------------------------------------------------
api.interceptors.request.use(
  (config: InternalAxiosRequestConfig): InternalAxiosRequestConfig => {
    // Proxy handling (prepend proxy URL)
    const proxyUrl = getProxyUrl()
    if (proxyUrl) {
      const originalUrl = `${config.baseURL}${config.url}`
      config.url = `${proxyUrl}${originalUrl}`
      config.baseURL = '' // prevent duplication
    }

    // Authentication token
    const token = localStorage.getItem('token')
    if (token) {
      config.headers.Authorization = `Token ${token}`
    }

    return config
  },
  (error: AxiosError) => Promise.reject(error)
)

// ----------------------------------------------------------------------
// Response interceptor: handle auth errors and global error formatting
// ----------------------------------------------------------------------
api.interceptors.response.use(
  (response: AxiosResponse): AxiosResponse => response,
  async (error: AxiosError<ApiErrorResponse>) => {
    const originalRequest = error.config as InternalAxiosRequestConfig & { _retry?: boolean }

    // Handle 401 Unauthorized
    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true
      localStorage.removeItem('token')
      // Avoid redirect loop if already on login page
      if (typeof window !== 'undefined' && !window.location.pathname.startsWith('/login')) {
        window.location.href = '/login'
      }
      return Promise.reject(error)
    }

    // Format error message for consistency
    const errorMessage =
      error.response?.data?.detail ||
      error.response?.data?.message ||
      error.response?.data?.error ||
      error.message ||
      'An unexpected error occurred'

    // Attach formatted message to error object for easier handling in components
    ;(error as any).userMessage = errorMessage

    return Promise.reject(error)
  }
)

// ----------------------------------------------------------------------
// Helper to create a cancelable request (AbortController)
// ----------------------------------------------------------------------
export const createCancelToken = () => {
  const controller = new AbortController()
  return { token: controller.signal, cancel: () => controller.abort() }
}

// ----------------------------------------------------------------------
// Type-safe request wrapper (optional, but convenient)
// ----------------------------------------------------------------------
export const request = {
  get: <T = any>(url: string, config?: AxiosRequestConfig) =>
    api.get<T>(url, config).then(res => res.data),
  post: <T = any>(url: string, data?: any, config?: AxiosRequestConfig) =>
    api.post<T>(url, data, config).then(res => res.data),
  put: <T = any>(url: string, data?: any, config?: AxiosRequestConfig) =>
    api.put<T>(url, data, config).then(res => res.data),
  patch: <T = any>(url: string, data?: any, config?: AxiosRequestConfig) =>
    api.patch<T>(url, data, config).then(res => res.data),
  delete: <T = any>(url: string, config?: AxiosRequestConfig) =>
    api.delete<T>(url, config).then(res => res.data),
}

// ----------------------------------------------------------------------
// Export the raw axios instance for advanced use cases
// ----------------------------------------------------------------------
export default api