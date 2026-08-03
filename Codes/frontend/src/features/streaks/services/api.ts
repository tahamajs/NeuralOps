import { request } from "../../../api/client";
import {
  StreakStats,
  WeeklySummary,
  HeatmapData,
  LeaderboardEntry,
} from "../types";

export const streaksAPI = {
  getHeatmap: async (
    username?: string,
    year?: number,
  ): Promise<HeatmapData> => ({
    year: year || new Date().getFullYear(),
    weeks: [],
    total: 0,
    range: { start: "", end: "" },
    streak_stats: {
      current_streak: 0,
      longest_streak: 0,
      total_active_days: 0,
      last_active_date: null,
      rank: null,
      streak_expires_at: null,
    },
  }),

  getMyStreak: async (): Promise<StreakStats> => ({
    current_streak: 0,
    longest_streak: 0,
    total_active_days: 0,
    last_active_date: null,
    rank: null,
    streak_expires_at: null,
  }),
  getUserStreak: async (username: string): Promise<StreakStats> => ({
    current_streak: 0,
    longest_streak: 0,
    total_active_days: 0,
    last_active_date: null,
    rank: null,
    streak_expires_at: null,
  }),

  getWeeklySummaries: async (
    username?: string,
    limit?: number,
  ): Promise<WeeklySummary[]> => [],
  getMyWeeklySummary: async (): Promise<WeeklySummary> =>
    ({} as WeeklySummary),
  getLeaderboard: async (): Promise<LeaderboardEntry[]> => [],
  generateWeeklySummary: async (): Promise<WeeklySummary> =>
    ({} as WeeklySummary),
  getDailyActivities: async (date?: string): Promise<any[]> => [],
};
