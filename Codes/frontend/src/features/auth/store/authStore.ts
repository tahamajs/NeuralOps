import { create } from "zustand";
import api from "../../../api/client";
import { ENDPOINTS } from "../../../api/endpoints";
import { getAuthToken, setAuthToken, removeAuthToken } from "../../../utils/authUtils";

interface AuthState {
  user: { username: string; email?: string } | null;
  isAuthenticated: boolean;
  login: (username: string, password: string) => Promise<void>;
  logout: () => void;
  register: (
    username: string,
    email: string,
    password: string,
  ) => Promise<void>;
  setAuth: (user: any, token: string) => void;
}

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  isAuthenticated: !!getAuthToken(),
  login: async (username, password) => {
    const res = await api.post(ENDPOINTS.AUTH.LOGIN, { username, password });
    if (res.data.token) {
      setAuthToken(res.data.token);
      set({ user: { username }, isAuthenticated: true });
    }
  },
  logout: () => {
    removeAuthToken();
    set({ user: null, isAuthenticated: false });
    window.location.href = "/login";
  },
  register: async (username, email, password) => {
    await api.post(ENDPOINTS.AUTH.REGISTER, { username, email, password });
  },
  setAuth: (user, token) => {
    set({ user, isAuthenticated: true });
  },
}));
