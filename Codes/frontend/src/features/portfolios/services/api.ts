import api from '../../../api/client'
import type { Portfolio, PortfolioStats, ShowcaseItem } from '../types'

export const portfolioAPI = {
  getPortfolio: (username: string) =>
    api.get(`/portfolios/user/${username}/`).then(res => res.data),

  getStats: (userId: string) =>
    api.get(`/portfolios/users/${userId}/stats/`).then(res => res.data),

  getShowcaseItems: (userId: string) =>
    api.get(`/portfolios/users/${userId}/showcase/`).then(res => res.data),

  getContributions: (userId: string, year?: number) =>
    api.get(`/portfolios/users/${userId}/contributions/`, { params: { year } })
      .then(res => res.data),

  follow: (username: string) =>
    api.post(`/portfolios/user/${username}/follow/`).then(res => res.data),

  unfollow: (username: string) =>
    api.post(`/portfolios/user/${username}/unfollow/`).then(res => res.data),

  getFollowers: (username: string) =>
    api.get(`/portfolios/user/${username}/followers/`).then(res => res.data),

  getFollowing: (username: string) =>
    api.get(`/portfolios/user/${username}/following/`).then(res => res.data),
}