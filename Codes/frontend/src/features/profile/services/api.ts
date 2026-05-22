import api from '../../../api/client'

export const profileAPI = {
  getProfile: (username?: string) =>
    username
      ? api.get(`/profiles/user/${username}/`).then(res => res.data)
      : api.get('/profiles/me/').then(res => res.data),

  updateProfile: (data: Partial<UserProfile>) =>
    api.patch('/profiles/me/', data).then(res => res.data),

  uploadAvatar: (file: File) => {
    const formData = new FormData()
    formData.append('avatar', file)
    return api.post('/profiles/me/avatar/', formData).then(res => res.data)
  },

  removeAvatar: () => api.delete('/profiles/me/avatar/remove/'),

  getUserStats: (userId: string) =>
    api.get(`/profiles/users/${userId}/stats/`).then(res => res.data),

  getUserActivity: (userId: string, page: number = 1) =>
    api.get(`/profiles/users/${userId}/activity/?page=${page}`).then(res => res.data),

  getPinnedItems: (userId: string) =>
    api.get(`/profiles/users/${userId}/pins/`).then(res => res.data),

  pinItem: (itemType: string, itemId: string) =>
    api.post('/profiles/me/pins/', { item_type: itemType, item_id: itemId }),

  unpinItem: (pinId: string) => api.delete(`/profiles/me/pins/${pinId}/`),

  getBadges: (userId: string) =>
    api.get(`/profiles/users/${userId}/badges/`).then(res => res.data),

  getOrganizationMemberships: (userId: string) =>
    api.get(`/profiles/users/${userId}/orgs/`).then(res => res.data),

  changePassword: (oldPassword: string, newPassword: string) =>
    api.post('/auth/change-password/', { old_password: oldPassword, new_password: newPassword }),
}