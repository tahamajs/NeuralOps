export type ExpertiseType = string
export type AvailabilityStatus = 'available' | 'busy' | 'looking'

export interface ExpertProfile {
  id: string
  user_id: string
  name: string
  title?: string
  avatar_url?: string
  institution: string
  department?: string
  expertise: ExpertiseType[]
  research_interests: string[]
  patents_count: number
  publications_count: number
  citations_count: number
  h_index: number
  availability: AvailabilityStatus
  linkedin_url?: string
  researchgate_url?: string
  orcid_id?: string
  location?: string
  bio?: string
  created_at: string
  updated_at: string
}

export interface Project {
  id: string
  title: string
  description: string
  required_expertise: ExpertiseType[]
  budget?: number
  timeline?: string
  status: 'open' | 'in_progress' | 'completed' | 'closed'
  created_by: string
  created_at: string
  updated_at: string
}

export interface Application {
  id: string
  project_id: string
  applicant_id: string
  status: 'pending' | 'accepted' | 'rejected'
  cover_letter?: string
  proposed_approach?: string
  expected_timeline?: string
  created_at: string
  updated_at: string
}

export interface FilterOptions {
  search?: string
  expertise?: ExpertiseType[]
  institution?: string
  availability?: AvailabilityStatus
  sort_by?: 'name' | 'reputation' | 'recent'
  page?: number
  limit?: number
}

export interface PaginatedResponse<T> {
  results: T[]
  count: number
  next: string | null
  previous: string | null
}