export type ProtocolStatus = 'draft' | 'published' | 'archived'
export type StepType = 'text' | 'image' | 'video' | 'resource' | 'decision'
export type ResourceType = 'dataset' | 'model' | 'code' | 'container' | 'external'

export interface ProtocolStep {
  id: string
  title: string
  description: string
  type: StepType
  order: number
  estimated_duration_minutes?: number
  resources?: ProtocolResource[]
}

export interface ProtocolResource {
  id: string
  name: string
  type: ResourceType
  url?: string
  version?: string
  required: boolean
}

export interface Protocol {
  id: string
  name: string
  description: string
  version: string
  status: ProtocolStatus
  steps: ProtocolStep[]
  tags: string[]
  author: string
  created_at: string
  updated_at: string
  usage_count: number
  derived_from?: string
  permissions: {
    can_edit: boolean
    can_delete: boolean
    can_publish: boolean
  }
}

export interface ProtocolVersion {
  version: string
  created_at: string
  author: string
  changes: string[]
  protocol_snapshot: Protocol
}