import { create } from "zustand";
import api from "../../../api/client";
import { ENDPOINTS } from "../../../api/endpoints";

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
  isAuthenticated: !!localStorage.getItem("token") || document.cookie.includes("authToken="),
  login: async (username, password) => {
    const res = await api.post(ENDPOINTS.AUTH.LOGIN, { username, password });
    if (res.data.token) {
      localStorage.setItem("token", res.data.token);
      document.cookie = `authToken=${res.data.token}; path=/; max-age=${60 * 60 * 24 * 30}; SameSite=Lax`;
      set({ user: { username }, isAuthenticated: true });
    }
  },
  logout: () => {
    localStorage.removeItem("token");
    document.cookie = "authToken=; path=/; max-age=0; SameSite=Lax";
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
