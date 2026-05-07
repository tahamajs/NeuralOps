export const API_ROUTES = {
  AUTH: { LOGIN: '/auth/login/', REGISTER: '/auth/register/' },
  EXPERIMENTS: { LIST: '/experiments/', DETAIL: (id: string) => `/experiments/${id}/` },
  // …
}