export type SuggestionReason = "expertise" | "interest" | "mutual" | "network";
export type MatchStatus = "pending" | "accepted" | "rejected" | "connected";
export type CollaborationType = "paper" | "grant" | "project" | "meeting";

export interface SuggestionScore {
  overall: number; // 0-100
  expertise: number; // 0-100
  interest: number; // 0-100
  network: number; // 0-100
  availability: number; // 0-100
}

export interface ResearchProfile {
  id: string;
  name: string;
  title?: string;
  avatar_url?: string;
  institution: string;
  department?: string;
  expertise: string[]; // e.g., ["Computer Vision", "Deep Learning"]
  research_interests: string[]; // more specific
  publications_count: number;
  citation_count: number;
  h_index: number;
  availability: "active" | "busy" | "looking";
  organization_id?: string;
}

export interface SuggestedCollaborator {
  id: string;
  profile: ResearchProfile;
  match_score: SuggestionScore;
  reason: SuggestionReason;
  common_connections: string[];
  common_interests: string[];
  mutual_followers: number;
  suggested_at: string;
}

export interface MatchHistoryEntry {
  id: string;
  collaborator: ResearchProfile;
  match_status: MatchStatus;
  collaboration_type: CollaborationType;
  started_at?: string;
  ended_at?: string;
  project_name?: string;
  outcome?: string;
  messages_exchanged: number;
  rating?: number;
  created_at: string;
  updated_at: string;
}

export interface MatchFilters {
  min_score?: number;
  expertise?: string[];
  institution?: string;
  availability?: string;
  sort_by?: "score" | "name" | "recent";
}

export interface PaginatedResponse<T> {
  results: T[];
  count: number;
  next: string | null;
  previous: string | null;
}
