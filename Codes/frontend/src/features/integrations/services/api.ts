import { request as api } from "../../../api/client";
import {
  CloudConnection,
  Webhook,
  WebhookDelivery,
  SandboxSession,
  Plugin,
} from "../types";

export const integrationsAPI = {
  // Cloud connections
  getConnections: () =>
    api.get<CloudConnection[]>("/integrations/cloud/connections/"),
  createConnection: (data: Partial<CloudConnection>) =>
    api.post<CloudConnection>("/integrations/cloud/connections/", data),
  deleteConnection: (id: string) =>
    api.delete(`/integrations/cloud/connections/${id}/`),
  testConnection: (id: string) =>
    api.post(`/integrations/cloud/connections/${id}/test/`),

  // Webhooks
  getWebhooks: () => api.get<Webhook[]>("/integrations/webhooks/"),
  createWebhook: (data: Partial<Webhook>) =>
    api.post<Webhook>("/integrations/webhooks/", data),
  updateWebhook: (id: string, data: Partial<Webhook>) =>
    api.patch<Webhook>(`/integrations/webhooks/${id}/`, data),
  deleteWebhook: (id: string) => api.delete(`/integrations/webhooks/${id}/`),
  testWebhook: (id: string) => api.post(`/integrations/webhooks/${id}/test/`),
  getWebhookDeliveries: (id: string) =>
    api.get<WebhookDelivery[]>(`/integrations/webhooks/${id}/deliveries/`),

  // Sandbox
  getSandboxSessions: () =>
    api.get<SandboxSession[]>("/integrations/sandbox/sessions/"),
  createSandboxSession: (environment: string) =>
    api.post<SandboxSession>("/integrations/sandbox/sessions/", {
      environment,
    }),
  terminateSandboxSession: (id: string) =>
    api.post(`/integrations/sandbox/sessions/${id}/terminate/`),
  getSandboxWebSocketUrl: (id: string) =>
    api.get<string>(`/integrations/sandbox/sessions/${id}/ws/`),

  // Plugin marketplace
  getPlugins: () => api.get<Plugin[]>("/integrations/plugins/"),
  installPlugin: (id: string) =>
    api.post(`/integrations/plugins/${id}/install/`),
  uninstallPlugin: (id: string) =>
    api.post(`/integrations/plugins/${id}/uninstall/`),
};
