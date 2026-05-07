import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],

  server: {
    port: 3000,
    host: '0.0.0.0',           // listen on all interfaces for Docker / LAN access

    // Proxy API + WebSocket requests to Django backend
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
        ws: true,               // enable WebSocket proxying for real-time features
      },
      '/ws': {
        target: 'ws://localhost:8000',
        changeOrigin: true,
        ws: true,
        secure: false,
      },
    },
  },

  build: {
    outDir: 'dist',
    sourcemap: false,           // disable for production (set to true for debugging)
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: true,     // remove console.log in production
        drop_debugger: true,
      },
    },
    rollupOptions: {
      output: {
        manualChunks: {
          // Split large dependencies into separate chunks
          vendor: ['react', 'react-dom', 'react-router-dom', '@tanstack/react-query'],
          charts: ['plotly.js-dist-min'],
        },
      },
    },
    chunkSizeWarningLimit: 1000,  // raise limit to avoid warnings for large chart bundles
  },
})