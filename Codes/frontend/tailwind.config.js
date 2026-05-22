/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        surface: '#0a0c10',
        'neon-blue': '#3b82f6',
      },
    },
  },
  plugins: [],
}