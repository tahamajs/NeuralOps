export type BadgeRarity = "common" | "rare" | "epic" | "legendary";
export type BadgeCategory =
  "experiments" | "notes" | "collaboration" | "streak" | "community";

export interface Badge {
  id: string;
  name: string;
  description: string;
  icon: string; // emoji or icon class
  rarity: BadgeRarity;
  category: BadgeCategory;
  earned_at?: string; // present only if earned
  progress_current?: number;
  progress_target?: number;
}

export interface BadgeStats {
  total: number;
  earned: number;
  by_rarity: Record<BadgeRarity, number>;
  by_category: Record<BadgeCategory, number>;
  next_badge?: Badge;
}

export interface LeaderboardEntry {
  rank: number;
  user_id: string;
  username: string;
  avatar_url?: string;
  total_points: number;
  level: number;
  tier: "bronze" | "silver" | "gold" | "platinum" | "diamond";
  change: number; // rank change vs previous period
}

export interface ContributionStats {
  total_points: number;
  level: number;
  points_to_next_level: number;
  tier: LeaderboardEntry["tier"];
  recent_contributions: {
    experiments: number;
    notes: number;
    comments: number;
    collaborations: number;
  };
}
