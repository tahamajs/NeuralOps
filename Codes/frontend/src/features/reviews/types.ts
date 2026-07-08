export type ReviewRequestStatus =
  "pending" | "in_progress" | "completed" | "expired";
export type ReviewStatus = "pending" | "submitted" | "accepted" | "rejected";

export interface User {
  id: string;
  username: string;
  avatar_url?: string;
  reputation: number;
}

export interface ReviewRequest {
  id: string;
  title: string;
  description: string;
  requester: User;
  reviewer?: User;
  status: ReviewRequestStatus;
  due_date: string;
  created_at: string;
  updated_at: string;
  tags: string[];
  priority: "low" | "medium" | "high";
  bounty: number;
  context: {
    experiment_id?: string;
    run_id?: string;
    paper_id?: string;
  };
}

export interface Review {
  id: string;
  request_id: string;
  reviewer: User;
  content: string;
  rating: number;
  status: ReviewStatus;
  created_at: string;
  updated_at: string;
}

export interface ReviewStats {
  total_requests: number;
  pending_reviews: number;
  completed_reviews: number;
  average_rating: number;
  total_bounty_earned: number;
}
