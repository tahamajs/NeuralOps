export type ModelStatus = 'draft' | 'pending' | 'approved' | 'rejected' | 'deployed'
export type Stage = 'development' | 'staging' | 'production' | 'archived'
export type ApprovalAction = 'approve' | 'reject' | 'request_changes'
export type Permission = 'view' | 'download' | 'deploy' | 'manage'

export interface Model {
  id: string
  name: string
  description?: string
  version: string
  framework: string
  task: string
  metrics: Record<string, number>
  status: ModelStatus
  stage: Stage
  created_by: string
  created_at: string
  updated_at: string
  tags: string[]
  artifact_url?: string
}

export interface ApprovalRequest {
  id: string
  model_id: string
  model_name: string
  model_version: string
  requested_by: string
  requested_at: string
  reviewers: string[]
  status: 'pending' | 'approved' | 'rejected'
  comments: ApprovalComment[]
}

export interface ApprovalComment {
  id: string
  user: string
  action: ApprovalAction
  comment: string
  created_at: string
}

export interface PromotionRequest {
  id: string
  model_id: string
  from_stage: Stage
  to_stage: Stage
  status: 'pending' | 'approved' | 'rejected'
  requested_by: string
  requested_at: string
  approved_by?: string
  approved_at?: string
}

export interface AccessPolicy {
  id: string
  model_id: string
  user_id: string
  username: string
  permission: Permission
  granted_by: string
  granted_at: string
  expires_at?: string
}

export interface CatalogSearchParams {
  q?: string
  framework?: string
  task?: string
  stage?: Stage
  tags?: string[]
  page?: number
  limit?: number
}

export interface PaginatedResponse<T> {
  results: T[]
  count: number
  next: string | null
  previous: string | null
}