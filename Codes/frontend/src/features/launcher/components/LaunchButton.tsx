import { Button } from '../../../components/ui/Button'

export const LaunchButton = ({ loading, onClick }: { loading: boolean; onClick: () => void }) => (
  <Button variant="primary" size="lg" className="w-full mt-4" disabled={loading} onClick={onClick}>
    {loading ? 'Launching...' : '🚀 Launch Experiment'}
  </Button>
)