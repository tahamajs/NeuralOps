import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  
  server: {
    port: 3000,
    host: '0.0.0.0', // Listen on all interfaces
    
    // Proxy API requests to Django backend
    proxy: {
      '/api': {
        target: process.env.VITE_API_URL || 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '/api'),
        secure: false,
        ws: true, // Enable WebSocket proxying
      },
      '/ws': {
        target: process.env.VITE_WS_URL || 'ws://localhost:8000',
        changeOrigin: true,
        ws: true,
        secure: false,
      },
    },
  },
  
  build: {
    outDir: 'dist',
    sourcemap: true,
    minify: 'terser',
  },
  
  define: {
    // Make environment variables available in client code
    'process.env.VITE_API_URL': JSON.stringify(process.env.VITE_API_URL || 'http://localhost:8000/api'),
    'process.env.VITE_WS_URL': JSON.stringify(process.env.VITE_WS_URL || 'ws://localhost:8000'),
  },
})
