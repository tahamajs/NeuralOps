import { Card } from '../../../components/ui/Card'

export default function ExperimentTimeline({ run }: { run: any }) {
  const events = [
    { date: run.created_at, label: 'Created', icon: '🚀', color: 'text-neon-blue' },
    { date: run.started_at, label: 'Started', icon: '▶️', color: 'text-neon-orange' },
    { date: run.completed_at, label: 'Completed', icon: '✅', color: 'text-neon-green' },
  ].filter(e => e.date)

  return (
    <Card>
      <h3 className="text-sm font-display font-bold text-white mb-4">Experiment Timeline</h3>
      <div className="space-y-4">
        {events.map((event, i) => (
          <div key={i} className="flex items-start gap-3">
            <div className={`w-2 h-2 rounded-full mt-1.5 ${event.color} bg-current`} />
            <div>
              <p className="text-sm text-white">{event.label}</p>
              <p className="text-xs text-slate-400">{new Date(event.date).toLocaleString()}</p>
            </div>
          </div>
        ))}
      </div>
    </Card>
  )
}