export type CloudProvider = 'aws' | 'gcp' | 'azure' | 'google_drive' | 'dropbox'
export type WebhookStatus = 'active' | 'inactive' | 'failed'
export type WebhookEvent = 'experiment.created' | 'experiment.completed' | 'run.failed' | 'model.deployed'

export interface CloudConnection {
  id: string
  provider: CloudProvider
  name: string
  credentials: {
    accessKey?: string
    secretKey?: string
    bucket?: string
    region?: string
    clientId?: string
    refreshToken?: string
  }
  status: 'connected' | 'disconnected' | 'error'
  createdAt: string
  lastUsed?: string
}

export interface Webhook {
  id: string
  name: string
  url: string
  events: WebhookEvent[]
  status: WebhookStatus
  secret?: string
  createdAt: string
  lastDelivery?: {
    status: number
    timestamp: string
  }
}

export interface WebhookDelivery {
  id: string
  webhookId: string
  event: WebhookEvent
  payload: any
  status: number
  response: string
  duration: number
  timestamp: string
}

export interface SandboxSession {
  id: string
  name: string
  status: 'active' | 'terminated' | 'timeout'
  createdAt: string
  lastActivity: string
  environment: 'python' | 'node' | 'bash' | 'r'
}

export interface Plugin {
  id: string
  name: string
  description: string
  version: string
  author: string
  icon?: string
  downloads: number
  rating: number
  installed: boolean
  repositoryUrl: string
  dependencies?: string[]
  permissions: string[]
}