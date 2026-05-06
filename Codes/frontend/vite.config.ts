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
                target: 'http://localhost:8000',
                changeOrigin: true,
                rewrite: (path) => path.replace(/^\/api/, '/api'),
                secure: false,
                ws: true, // Enable WebSocket proxying
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
        sourcemap: false,
        minify: 'terser',
        terserOptions: {
            compress: {
                drop_console: true,
                drop_debugger: true,
            },
        },
        rollupOptions: {
            output: {
                manualChunks: {
                    vendor: ['react', 'react-dom', 'react-router-dom', '@tanstack/react-query'],
                    charts: ['plotly.js-dist-min'],
                },
            },
        },
        chunkSizeWarningLimit: 1000,
    },
})
