import api from '../../../api/client'
import { Badge, BadgeStats, LeaderboardEntry, ContributionStats } from '../types'

export const gamificationAPI = {
  // Badges
  getBadges: () => api.get<Badge[]>('/gamification/badges/'),
  getMyBadges: () => api.get<Badge[]>('/gamification/badges/mine/'),
  getBadgeStats: () => api.get<BadgeStats>('/gamification/badges/stats/'),

  // Leaderboard
  getLeaderboard: (params?: { period?: 'weekly' | 'monthly' | 'all'; limit?: number }) =>
    api.get<LeaderboardEntry[]>('/gamification/leaderboard/', { params }),

  // User stats
  getMyStats: () => api.get<ContributionStats>('/gamification/stats/me/'),
}