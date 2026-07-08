export interface Organization {
  id: string;
  name: string;
  slug: string;
  description?: string;
  avatar_url?: string;
  created_at: string;
  updated_at: string;
  member_count: number;
  is_owner: boolean;
  is_billing_contact: boolean;
  current_user_role: "owner" | "billing_contact" | "member";
  subscription_plan?: SubscriptionPlan;
  subscription_status?: "active" | "trialing" | "past_due" | "canceled";
}

export interface OrganizationMember {
  id: string;
  user_id: string;
  username: string;
  email: string;
  avatar_url?: string;
  role: "owner" | "billing_contact" | "member";
  joined_at: string;
}

export interface Team {
  id: string;
  name: string;
  description?: string;
  member_count: number;
  members: TeamMember[];
}

export interface TeamMember {
  user_id: string;
  username: string;
  role: "lead" | "member";
}

export interface Invitation {
  id: string;
  email: string;
  role: "owner" | "billing_contact" | "member";
  invited_by: string;
  created_at: string;
  expires_at: string;
  status: "pending" | "accepted" | "expired";
}

export interface SubscriptionPlan {
  id: string;
  name: string;
  price_monthly: number;
  price_yearly: number;
  features: string[];
  member_limit: number;
  team_limit: number;
  storage_limit_gb: number;
}

export interface Invoice {
  id: string;
  amount: number;
  currency: string;
  status: "paid" | "pending" | "failed";
  period_start: string;
  period_end: string;
  pdf_url: string;
}

export interface AuditLogEntry {
  id: string;
  action: string;
  user: string;
  details: Record<string, any>;
  ip_address: string;
  created_at: string;
}

export interface ResourceGroup {
  id: string;
  name: string;
  description?: string;
  experiment_count: number;
  created_at: string;
}
