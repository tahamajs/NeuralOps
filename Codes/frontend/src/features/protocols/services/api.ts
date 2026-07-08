import api from "../../../api/client";
import { Protocol, ProtocolVersion } from "../types";

export const protocolsAPI = {
  // List protocols
  getProtocols: (params?: {
    search?: string;
    tags?: string[];
    status?: string;
  }) =>
    api.get<{ results: Protocol[]; count: number }>("/protocols/", { params }),

  // Single protocol
  getProtocol: (id: string) => api.get<Protocol>(`/protocols/${id}/`),

  // Create / Update / Delete
  createProtocol: (data: Partial<Protocol>) =>
    api.post<Protocol>("/protocols/", data),
  updateProtocol: (id: string, data: Partial<Protocol>) =>
    api.patch(`/protocols/${id}/`, data),
  deleteProtocol: (id: string) => api.delete(`/protocols/${id}/`),

  // Versioning
  getVersionHistory: (id: string) =>
    api.get<ProtocolVersion[]>(`/protocols/${id}/versions/`),
  revertToVersion: (id: string, version: string) =>
    api.post(`/protocols/${id}/revert/`, { version }),

  // Publishing
  publishProtocol: (id: string) => api.post(`/protocols/${id}/publish/`),
  archiveProtocol: (id: string) => api.post(`/protocols/${id}/archive/`),
};
