import { Link } from 'react-router-dom'
import { Button } from '../components/ui/Button'
import { Compass } from 'lucide-react'

export default function NotFoundPage() {
  return (
    <div className="min-h-[80vh] flex flex-col items-center justify-center text-center">
      {/* Glow effect behind the 404 */}
      <div className="relative">
        <div className="absolute inset-0 bg-neon-blue/20 blur-3xl rounded-full scale-150" />
        <h1 className="relative text-8xl font-display font-black text-white mb-4 animate-glow-pulse">
          404
        </h1>
      </div>

      <div className="flex items-center gap-2 text-slate-400 mb-2">
        <Compass size={20} className="text-neon-blue" />
        <p className="text-lg font-medium">Lost in the neural space</p>
      </div>

      <p className="text-slate-500 text-sm mb-8 max-w-md">
        The page you're looking for doesn't exist or has been moved.
        Let's get you back on track.
      </p>

      <div className="flex gap-3">
        <Link to="/dashboard">
          <Button variant="primary" size="md">
            ⬅ Back to Dashboard
          </Button>
        </Link>
        <Button variant="ghost" size="md" onClick={() => window.history.back()}>
          Go Back
        </Button>
      </div>

      <p className="text-[10px] text-slate-600 mt-8">
        If you believe this is an error, contact your administrator.
      </p>
    </div>
  )
}