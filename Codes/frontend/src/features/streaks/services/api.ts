import { request } from "../../../api/client";
import { StreakStats, WeeklySummary, HeatmapData, LeaderboardEntry } from "../types";

export const streaksAPI = {
  getHeatmap: async (username?: string, year?: number) =>
    ({ data: { current_streak: 0, max_streak: 0, dates: {} } }),

  getMyStreak: async () => ({ data: { username: "me", current_streak: 0, longest_streak: 0, last_activity: "" } }),
  getUserStreak: async (username: string) => ({ data: { username, current_streak: 0, longest_streak: 0, last_activity: "" } }),

  getWeeklySummaries: async (username?: string, limit?: number) => ({ data: [] }),
  getMyWeeklySummary: async () => ({ data: {} }),
  getLeaderboard: async () => ({ data: [] }),
  generateWeeklySummary: async () => ({ data: {} }),
  getDailyActivities: async (date?: string) => ({ data: [] }),
};
