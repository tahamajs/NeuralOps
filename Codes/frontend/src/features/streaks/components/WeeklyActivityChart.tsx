import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { TerminalCard } from '../../../components/ui/TerminalCard'
import { WeeklySummary } from '../types'
import { startOfWeek, eachDayOfInterval, format, addDays } from 'date-fns'

export function WeeklyActivityChart({ summary }: { summary: WeeklySummary }) {
  const start = startOfWeek(new Date(summary.week_start), { weekStartsOn: 1 })
  const days = eachDayOfInterval({ start, end: addDays(start, 6) })
  const chartData = days.map(day => ({ day: format(day, 'EEE'), value: summary.breakdown.experiments + summary.breakdown.notes }))

  return (
    <TerminalCard className="p-4"><h3 className="text-sm font-bold text-white mb-3">Daily Activity Pattern</h3><ResponsiveContainer width="100%" height={200}><BarChart data={chartData}><XAxis dataKey="day" stroke="#475569" /><YAxis stroke="#475569" /><Tooltip contentStyle={{ backgroundColor: '#0a0f1a' }} /><Bar dataKey="value" radius={[4, 4, 0, 0]}><Cell fill="#3b82f6" /></Bar></BarChart></ResponsiveContainer></TerminalCard>
  )
}