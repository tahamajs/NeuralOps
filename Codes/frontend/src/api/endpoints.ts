export const ENDPOINTS = {
  AUTH: {
    LOGIN: '/auth/login/',
    REGISTER: '/auth/register/',
    LOGOUT: '/auth/logout/',
  },
  EXPERIMENTS: {
    LIST: '/experiments/',
    DETAIL: (id: string) => `/experiments/${id}/`,
    ARTIFACTS: (id: string) => `/experiments/${id}/artifacts/`,
    MODEL: (id: string) => `/experiments/${id}/model/`,
    PAPER: (id: string) => `/experiments/${id}/paper/`,
    TRAIN: '/train/',
    EVALUATE: '/evaluate/',
    COMPRESS: '/compress/',
  },
  CONFIG: {
    MODELS: '/config/models/',
    TASKS: '/config/tasks/',
    ATTENTIONS: '/config/attentions/',
    METHODS: '/config/methods/',
  },
  NOTES: {
  LIST: '/notes/v1/notes/',
  DETAIL: (id: string) => `/notes/v1/notes/${id}/`,
  REACT: (id: string) => `/notes/v1/notes/${id}/react/`,
  VERSIONS: (id: string) => `/notes/v1/notes/${id}/versions/`,
  SHARE: (id: string) => `/notes/v1/notes/${id}/share/`,
  RUN_NOTES: (runId: string) => `/notes/v1/notes/runs/${runId}/`,
  REPLIES: (id: string) => `/notes/v1/notes/${id}/replies/`,
  MENTIONS: (id: string) => `/notes/v1/notes/${id}/mentions/`,
  EXPORT_PDF: (id: string) => `/notes/v1/notes/${id}/export/pdf/`,
  STATISTICS: '/notes/v1/notes/statistics/',
  RECENT: '/notes/v1/notes/recent/',
  BULK: '/notes/v1/notes/bulk/',
  BULK_TAGS: '/notes/v1/notes/bulk/tags/',
  WEBHOOKS: '/notes/v1/webhooks/',
},
  SYSTEM: {
    STATUS: '/system/status/',
    GPU: '/system/gpu/',
  },
  HEALTH: '/health/',
  VERSION: '/version/',



// src/api/endpoints.ts  (add inside the exported ENDPOINTS object)

  MARKETPLACE: {
    LISTINGS: '/marketplace/listings/',
    ORDERS: '/marketplace/orders/',
    JOBS: '/marketplace/jobs/',
    JOB_DETAIL: (id: string) => `/marketplace/jobs/${id}/`,
    BALANCE: '/marketplace/balance/',
    INVOICES: '/marketplace/invoices/',
    USAGE: '/marketplace/usage/',
    LEADERBOARD: '/marketplace/leaderboard/',
    REPUTATION: '/marketplace/reputation/',
  },
}