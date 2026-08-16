// src/types/user.ts
export interface User {
  id: number;
  username: string;
  email: string;
  first_name?: string;
  last_name?: string;
  is_staff?: boolean;
  date_joined?: string;
  last_login?: string | null;
}

export interface UserProfile {
  id: string | number;
  username: string;
  email: string;
  display_name?: string;
  full_name?: string;
  bio?: string;
  website?: string;
  location?: string;
  avatar?: string | null;
  avatar_url?: string;
  affiliation?: string;
  research_interests?: string[];
  skills?: string[];
  github?: string;
  twitter?: string;
  linkedin?: string;
  google_scholar?: string;
  orcid?: string;
  total_experiments?: number;
  total_compressions?: number;
  total_papers?: number;
  date_joined?: string;
  created_at?: string;
  updated_at?: string;
}

export interface AuthResponse {
  token: string;
  username: string;
  email: string;
  user_id: number;
  is_staff?: boolean;
}
