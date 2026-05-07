export const ENDPOINTS = {
  AUTH: { LOGIN: '/auth/login/', REGISTER: '/auth/register/', LOGOUT: '/auth/logout/' },
  EXPERIMENTS: {
    LIST: '/experiments/',
    DETAIL: (id: string) => `/experiments/${id}/`,
    ARTIFACTS: (id: string) => `/experiments/${id}/artifacts/`,
    MODEL: (id: string) => `/experiments/${id}/model/`,
  },
  CONFIG: {
    MODELS: '/config/models/',
    TASKS: '/config/tasks/',
    ATTENTIONS: '/config/attentions/',
    METHODS: '/config/methods/',
  },
  SYSTEM: { STATUS: '/system/status/', GPU: '/system/gpu/' },
  HEALTH: '/health/',
  VERSION: '/version/',
};