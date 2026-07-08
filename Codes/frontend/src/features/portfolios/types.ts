export interface Portfolio {
  id: string;
  username: string;
  full_name: string;
  avatar_url: string | null;
  bio: string;
  location: string | null;
  website: string | null;
  company: string | null;
  github_username: string | null;
  twitter_username: string | null;
  linkedin_url: string | null;
  expertise: string[];
  research_interests: string[];
  created_at: string;
  updated_at: string;
  is_following: boolean;
  followers_count: number;
  following_count: number;
}

export interface PortfolioStats {
  total_experiments: number;
  total_runs: number;
  total_models_published: number;
  total_papers: number;
  total_notes: number;
  total_upvotes_received: number;
  reputation_score: number;
  rank: number;
  total_contributions: number;
  longest_streak: number;
  current_streak: number;
}

export interface ShowcaseItem {
  id: string;
  type: "experiment" | "paper" | "model" | "note";
  title: string;
  description: string;
  thumbnail_url?: string;
  url: string;
  created_at: string;
  stats?: Record<string, unknown>;
  tags?: string[];
}

export interface SocialLink {
  platform: "github" | "twitter" | "linkedin" | "website";
  url: string;
  username?: string;
}
