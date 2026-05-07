export const ENDPOINTS = {
  ...,
  NOTES: {
    LIST: '/notes/v1/notes/',
    DETAIL: (id: string) => `/notes/v1/notes/${id}/`,
    REACT: (id: string) => `/notes/v1/notes/${id}/react/`,
    VERSIONS: (id: string) => `/notes/v1/notes/${id}/versions/`,
    RUN_NOTES: (runId: string) => `/notes/v1/notes/runs/${runId}/`,
    STATISTICS: '/notes/v1/notes/statistics/',
    RECENT: '/notes/v1/notes/recent/',
    BULK: '/notes/v1/notes/bulk/',
    WEBHOOKS: '/notes/v1/webhooks/',
  }
}