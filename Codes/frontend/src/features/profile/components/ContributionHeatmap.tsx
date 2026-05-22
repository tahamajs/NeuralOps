import { useQuery } from '@tanstack/react-query'
import api from '../../../api/client'
import { Spinner } from '../../../components/ui/Spinner'

interface ContributionDay {
  date: string
  count: number
  level: 0 | 1 | 2 | 3 | 4
}

interface ContributionHeatmapProps {
  userId: string
  year?: number
}

export function ContributionHeatmap({ userId, year = new Date().getFullYear() }: ContributionHeatmapProps) {
  const { data, isLoading } = useQuery({
    queryKey: ['contribution-calendar', userId, year],
    queryFn: async () => (await api.get(`/profiles/users/${userId}/contributions/?year=${year}`)).data,
  })

  if (isLoading) return <Spinner size="sm" />

  const weeks = data?.weeks || []
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

  const getColor = (level: number) => {
    switch (level) {
      case 1: return 'bg-green-900'
      case 2: return 'bg-green-700'
      case 3: return 'bg-green-500'
      case 4: return 'bg-green-300'
      default: return 'bg-white/5'
    }
  }

  return (
    <div className="overflow-x-auto">
      <div className="flex gap-1">
        {weeks.map((week: ContributionDay[], weekIdx: number) => (
          <div key={weekIdx} className="flex flex-col gap-1">
            {week.map((day, dayIdx) => (
              <div
                key={dayIdx}
                className={`w-3 h-3 rounded-sm ${getColor(day.level)}`}
                title={`${day.date}: ${day.count} contributions`}
              />
            ))}
          </div>
        ))}
      </div>
      <div className="flex justify-between mt-2 text-xs text-slate-500">
        {months.map(month => <span key={month}>{month}</span>)}
      </div>
    </div>
  )
}