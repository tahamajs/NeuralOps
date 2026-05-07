import { Link } from 'react-router-dom'
import { Button } from '../components/ui/Button'

export default function NotFoundPage() {
  return (
    <div className="min-h-[80vh] flex flex-col items-center justify-center text-center">
      <h1 className="text-6xl font-display font-black text-white mb-4">404</h1>
      <p className="text-slate-400 mb-8">This page doesn't exist in the neural space.</p>
      <Link to="/dashboard">
        <Button>Back to Dashboard</Button>
      </Link>
    </div>
  )
}