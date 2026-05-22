import { useState, useEffect } from 'react'
import { Sun, Moon } from 'lucide-react'
import { useLocalStorage } from '../../../hooks/useLocalStorage'

export function ThemeToggle() {
  const [theme, setTheme] = useLocalStorage<'dark' | 'light'>('theme', 'dark')
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
    const root = document.documentElement
    if (theme === 'dark') {
      root.classList.add('dark')
    } else {
      root.classList.remove('dark')
    }
  }, [theme])

  if (!mounted) return null

  return (
    <button
      onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
      className="p-2 rounded-lg hover:bg-white/5 transition-colors"
      aria-label="Toggle theme"
    >
      {theme === 'dark' ? <Sun size={18} className="text-slate-400" /> : <Moon size={18} className="text-slate-400" />}
    </button>
  )
}