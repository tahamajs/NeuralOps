import { request } from "../../../api/client";
import { StreakStats, WeeklySummary, HeatmapData, LeaderboardEntry } from "../types";

export const streaksAPI = {
  getHeatmap: (username?: string, year?: number) =>
    request.get<HeatmapData>("/streaks/heatmap/", { params: { username, year } }),

  getMyStreak: () => request.get<StreakStats>("/streaks/streak/me/"),
  getUserStreak: (username: string) =>
    request.get<StreakStats>(`/streaks/streak/${username}/`),

  getWeeklySummaries: (username?: string, limit?: number) =>
    request.get<WeeklySummary[]>("/streaks/weekly/me/", {
      params: { username, limit },
    }),

  getMyWeeklySummary: () =>
    request.get<WeeklySummary>("/streaks/weekly/me/latest/"),

  getLeaderboard: () => request.get<LeaderboardEntry[]>("/streaks/leaderboard/"),

  generateWeeklySummary: () => request.post("/streaks/weekly/generate/"),

  getDailyActivities: (date?: string) =>
    request.get("/streaks/daily/me/", { params: { date } }),
};
