import { request } from "../../../api/client";
import {
  UserProfile,
  UserStats,
  UserActivity,
  PinnedItem,
  Badge,
  OrganizationMembership,
} from "../types";

export type {
  UserProfile,
  UserStats,
  UserActivity,
  PinnedItem,
  Badge,
  OrganizationMembership,
};

export const profileAPI = {
  getProfile: (username?: string) =>
    username
      ? request.get<UserProfile>(`/profiles/user/${username}/`)
      : request.get<UserProfile>("/profiles/me/"),

  updateProfile: (data: Partial<UserProfile>) =>
    request.patch<UserProfile>("/profiles/me/", data),

  uploadAvatar: (file: File) => {
    const formData = new FormData();
    formData.append("avatar", file);
    return request.post<UserProfile>("/profiles/me/avatar/", formData);
  },

  removeAvatar: () => request.delete("/profiles/me/avatar/remove/"),

  getUserStats: (userId: string) =>
    request.get<UserStats>(`/profiles/users/${userId}/stats/`),

  getUserActivity: (userId: string, page: number = 1) =>
    request.get<{ results: UserActivity[]; next: string | null; count: number }>(
      `/profiles/users/${userId}/activity/`,
      { params: { page } },
    ),

  getPinnedItems: (userId: string) =>
    request.get<PinnedItem[]>(`/profiles/users/${userId}/pins/`),

  pinItem: (itemType: string, itemId: string) =>
    request.post("/profiles/me/pins/", { item_type: itemType, item_id: itemId }),

  unpinItem: (pinId: string) => request.delete(`/profiles/me/pins/${pinId}/`),

  getBadges: (userId: string) =>
    request.get<Badge[]>(`/profiles/users/${userId}/badges/`),

  getOrganizationMemberships: (userId: string) =>
    request.get<OrganizationMembership[]>(`/profiles/users/${userId}/orgs/`),

  changePassword: (oldPassword: string, newPassword: string) =>
    request.post("/auth/change-password/", {
      old_password: oldPassword,
      new_password: newPassword,
    }),
};
