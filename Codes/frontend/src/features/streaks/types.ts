export interface ContributionDay {
  date: string;
  count: number;
  level: 0 | 1 | 2 | 3 | 4;
}

export interface StreakStats {
  current_streak: number;
  longest_streak: number;
  total_active_days: number;
  last_active_date: string | null;
  rank: number | null;
  streak_expires_at: string | null;
}

export interface WeeklySummary {
  week_start: string;
  week_end: string;
  total_contributions: number;
  average_daily: number;
  best_day: { date: string; count: number };
  top_experiment?: string;
  completed_tasks: number;
  streak_maintained: boolean;
  insights: string[];
  breakdown: {
    experiments: number;
    notes: number;
    comments: number;
    collaborations: number;
  };
}

export interface HeatmapData {
  year: number;
  weeks: ContributionDay[][];
  total: number;
  range: { start: string; end: string };
  streak_stats: StreakStats;
}

export interface LeaderboardEntry {
  user_id: string;
  username: string;
  current_streak: number;
  longest_streak: number;
  total_active_days: number;
}
