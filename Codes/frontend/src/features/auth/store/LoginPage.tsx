import { Link } from 'react-router-dom'
import { Card } from '../../../components/ui/Card'
import { Button } from '../../../components/ui/Button'

export default function LoginPage() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-surface">
      <Card className="p-8 w-96 text-center">
        <h1 className="text-2xl font-bold text-white mb-4">Login</h1>
        <Link to="/dashboard">
          <Button variant="primary">Go to Dashboard</Button>
        </Link>
      </Card>
    </div>
  )
}