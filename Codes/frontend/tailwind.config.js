/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        cyber: {
          950: '#04070e',
          900: '#0a0c10',
          800: '#141820',
          700: '#1e2330',
          600: '#2a3040',
          500: '#3f4a60',
          400: '#6b7b9a',
          300: '#8ca0c0',
          200: '#b6c5e0',
          100: '#dce3f0',
        },
        neon: {
          blue: '#3b82f6',
          purple: '#8b5cf6',
          pink: '#ec4899',
          green: '#10b981',
          orange: '#f59e0b',
          red: '#ef4444',
          cyan: '#06b6d4',
        },
        surface: {
          DEFAULT: '#0f172a',
          light: '#1e293b',
          dark: '#020617',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
        display: ['Manrope', 'system-ui', 'sans-serif'],
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-out',
        'slide-up': 'slideUp 0.4s ease-out',
        'slide-down': 'slideDown 0.3s ease-out',
        'scale-in': 'scaleIn 0.2s ease-out',
        'glow-pulse': 'glowPulse 2s ease-in-out infinite',
        'float': 'float 3s ease-in-out infinite',
        'shimmer': 'shimmer 2s linear infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        slideDown: {
          '0%': { opacity: '0', transform: 'translateY(-10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        scaleIn: {
          '0%': { opacity: '0', transform: 'scale(0.95)' },
          '100%': { opacity: '1', transform: 'scale(1)' },
        },
        glowPulse: {
          '0%, 100%': { boxShadow: '0 0 20px rgba(59,130,246,0.3)' },
          '50%': { boxShadow: '0 0 40px rgba(59,130,246,0.6)' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-5px)' },
        },
        shimmer: {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
      },
      backgroundImage: {
        'glow-conic':
          'conic-gradient(from 0deg at 50% 50%, var(--tw-gradient-stops))',
        'shimmer-gradient':
          'linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.05) 50%, transparent 100%)',
      },
      boxShadow: {
        'glow-sm': '0 0 10px rgba(59,130,246,0.2)',
        'glow-md': '0 0 20px rgba(59,130,246,0.3)',
        'glow-lg': '0 0 40px rgba(59,130,246,0.4)',
        'inner-glow': 'inset 0 0 20px rgba(59,130,246,0.1)',
      },
    },
  },
  plugins: [],
};