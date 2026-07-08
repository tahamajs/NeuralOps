import { request } from "../../../api/client";
import { Protocol, ProtocolVersion } from "../types";

export const protocolsAPI = {
  // List protocols
  getProtocols: (params?: {
    search?: string;
    tags?: string[];
    status?: string;
  }) =>
    request.get<{ results: Protocol[]; count: number }>("/protocols/", { params }),

  // Single protocol
  getProtocol: (id: string) => request.get<Protocol>(`/protocols/${id}/`),

  // Create / Update / Delete
  createProtocol: (data: Partial<Protocol>) =>
    request.post<Protocol>("/protocols/", data),
  updateProtocol: (id: string, data: Partial<Protocol>) =>
    request.patch(`/protocols/${id}/`, data),
  deleteProtocol: (id: string) => request.delete(`/protocols/${id}/`),

  // Versioning
  getVersionHistory: (id: string) =>
    request.get<ProtocolVersion[]>(`/protocols/${id}/versions/`),
  revertToVersion: (id: string, version: string) =>
    request.post(`/protocols/${id}/revert/`, { version }),

  // Publishing
  publishProtocol: (id: string) => request.post(`/protocols/${id}/publish/`),
  archiveProtocol: (id: string) => request.post(`/protocols/${id}/archive/`),
};
