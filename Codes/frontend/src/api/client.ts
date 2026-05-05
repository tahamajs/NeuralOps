import axios, { AxiosError, AxiosResponse } from 'axios'

// API client configuration
const apiBaseURL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'

const api = axios.create({
  baseURL: apiBaseURL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
  withCredentials: true, // Include cookies in requests
})

// Request interceptor: Attach token if available
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor: Handle auth errors
api.interceptors.response.use(
  (response) => {
    // Store token from response if provided
    if (response.data?.token) {
      localStorage.setItem('token', response.data.token)
    }
    return response
  },
  (error: AxiosError) => {
    if (error.response?.status === 401) {
      // Clear token on unauthorized
      localStorage.removeItem('token')
      // Redirect to login if in browser
      if (typeof window !== 'undefined') {
        window.location.href = '/login'
      }
    }
    return Promise.reject(error)
  }
)

export default api
export type { AxiosError, AxiosResponse }