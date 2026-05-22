import api from '../../../api/client'
import {
  Model,
  ApprovalRequest,
  PromotionRequest,
  AccessPolicy,
  CatalogSearchParams,
  PaginatedResponse,
} from '../types'

export const modelHubAPI = {
  // Model registry
  getModels: (params?: CatalogSearchParams) =>
    api.get<PaginatedResponse<Model>>('/model-hub/models/', { params }),
  getModel: (id: string) => api.get<Model>(`/model-hub/models/${id}/`),
  createModel: (data: Partial<Model>) => api.post<Model>('/model-hub/models/', data),
  updateModel: (id: string, data: Partial<Model>) => api.patch(`/model-hub/models/${id}/`, data),
  deleteModel: (id: string) => api.delete(`/model-hub/models/${id}/`),

  // Approval workflows
  getApprovalRequests: (params?: { status?: string }) =>
    api.get<ApprovalRequest[]>('/model-hub/approvals/', { params }),
  getApprovalRequest: (id: string) => api.get<ApprovalRequest>(`/model-hub/approvals/${id}/`),
  submitApproval: (id: string, action: ApprovalAction, comment?: string) =>
    api.post(`/model-hub/approvals/${id}/review/`, { action, comment }),

  // Promotion
  getPromotionRequests: () => api.get<PromotionRequest[]>('/model-hub/promotions/'),
  requestPromotion: (modelId: string, toStage: Stage) =>
    api.post('/model-hub/promotions/', { model_id: modelId, to_stage: toStage }),
  approvePromotion: (requestId: string) =>
    api.post(`/model-hub/promotions/${requestId}/approve/`),
  rejectPromotion: (requestId: string) =>
    api.post(`/model-hub/promotions/${requestId}/reject/`),

  // Access control
  getAccessPolicies: (modelId: string) =>
    api.get<AccessPolicy[]>(`/model-hub/models/${modelId}/access/`),
  grantAccess: (modelId: string, userId: string, permission: Permission, expiresAt?: string) =>
    api.post(`/model-hub/models/${modelId}/access/`, { user_id: userId, permission, expires_at: expiresAt }),
  revokeAccess: (policyId: string) => api.delete(`/model-hub/access/${policyId}/`),

  // Catalog search (full‑text search)
  searchCatalog: (params: CatalogSearchParams) =>
    api.get<PaginatedResponse<Model>>('/model-hub/catalog/search/', { params }),
}