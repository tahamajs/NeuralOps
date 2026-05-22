import api from '../../../api/client'
import { StreakStats, WeeklySummary, HeatmapData } from '../types'

export const streaksAPI = {
  getHeatmap: (username?: string, year?: number) =>
    api.get<HeatmapData>('/streaks/heatmap/', { params: { username, year } }),

  getMyStreak: () => api.get<StreakStats>('/streaks/streak/me/'),
  getUserStreak: (username: string) => api.get<StreakStats>(`/streaks/streak/${username}/`),

  getWeeklySummaries: (username?: string, limit?: number) =>
    api.get<WeeklySummary[]>('/streaks/weekly/me/', { params: { username, limit } }),

  getMyWeeklySummary: () => api.get<WeeklySummary>('/streaks/weekly/me/latest/'),

  getLeaderboard: () => api.get<StreakStats[]>('/streaks/leaderboard/'),

  generateWeeklySummary: () => api.post('/streaks/weekly/generate/'),

  getDailyActivities: (date?: string) =>
    api.get('/streaks/daily/me/', { params: { date } }),
}