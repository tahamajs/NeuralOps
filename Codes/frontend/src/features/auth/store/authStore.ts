import { create } from "zustand";
import api from "../../../api/client";
import { ENDPOINTS } from "../../../api/endpoints";
import {
  getAuthToken,
  setAuthToken,
  removeAuthToken,
} from "../../../utils/authUtils";

import { User } from "../../../types/user";

interface AuthState {
  user: Partial<User> | null;
  token: string | null;
  isAuthenticated: boolean;
  login: (username: string, password: string) => Promise<void>;
  logout: () => void;
  register: (
    username: string,
    email: string,
    password: string,
  ) => Promise<void>;
  setAuth: (user: any, token?: string) => void;
}

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  token: getAuthToken() || null,
  isAuthenticated: !!getAuthToken(),
  login: async (username, password) => {
    const res = await api.post(ENDPOINTS.AUTH.LOGIN, { username, password });
    if (res.data.token) {
      setAuthToken(res.data.token);
      set({
        user: {
          id: res.data.user_id,
          username: res.data.username || username,
          email: res.data.email,
          is_staff: res.data.is_staff,
        },
        token: res.data.token,
        isAuthenticated: true,
      });
    }
  },
  logout: () => {
    removeAuthToken();
    set({ user: null, token: null, isAuthenticated: false });
    window.location.href = "/login";
  },
  register: async (username, email, password) => {
    await api.post(ENDPOINTS.AUTH.REGISTER, { username, email, password });
  },
  setAuth: (user, token) => {
    if (token) {
      setAuthToken(token);
    }
    const currentToken = token || getAuthToken() || null;
    set({ user, token: currentToken, isAuthenticated: !!currentToken });
  },
}));
