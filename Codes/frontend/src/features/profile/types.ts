export interface UserProfile {
  id: string;
  username: string;
  email: string;
  full_name: string;
  avatar_url: string | null;
  bio: string;
  location: string;
  website: string;
  company: string;
  github_username: string;
  twitter_username: string;
  linkedin_url: string;
  affiliation: string;
  expertise: string[]; // e.g., ["Computer Vision", "NLP", "Reinforcement Learning"]
  research_interests: string[];
  pronouns: string;
  timezone: string;
  created_at: string;
  updated_at: string;
  is_active: boolean;
  is_staff: boolean;
}

export interface UserStats {
  total_experiments: number;
  total_runs: number;
  total_gpu_hours: number;
  total_models_published: number;
  total_papers: number;
  total_notes: number;
  total_questions: number;
  total_answers: number;
  total_upvotes_received: number;
  reputation_score: number;
  rank: number;
  contribution_days: number;
  longest_streak: number;
  current_streak: number;
}

export interface UserActivity {
  id: string;
  action: string;
  target_type: "experiment" | "run" | "note" | "paper" | "question" | "answer";
  target_id: string;
  target_title: string;
  created_at: string;
}

export interface PinnedItem {
  id: string;
  type: "experiment" | "note" | "paper";
  item_id: string;
  title: string;
  description: string;
  created_at: string;
}

export interface Badge {
  id: string;
  name: string;
  description: string;
  icon_url: string;
  awarded_at: string;
}

export interface OrganizationMembership {
  id: string;
  name: string;
  slug: string;
  avatar_url: string;
  role: "owner" | "billing_contact" | "member";
}
