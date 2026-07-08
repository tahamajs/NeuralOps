import { request } from "../../../api/client";
import {
  Model,
  ApprovalRequest,
  PromotionRequest,
  AccessPolicy,
  CatalogSearchParams,
  PaginatedResponse,
  ApprovalAction,
  Stage,
  Permission,
} from "../types";

export const modelHubAPI = {
  // Model registry
  getModels: (params?: CatalogSearchParams) =>
    request.get<PaginatedResponse<Model>>("/model-hub/models/", { params }),

  getModel: (id: string) => request.get<Model>(`/model-hub/models/${id}/`),

  createModel: (data: Partial<Model>) =>
    request.post<Model>("/model-hub/models/", data),

  updateModel: (id: string, data: Partial<Model>) =>
    request.patch<Model>(`/model-hub/models/${id}/`, data),

  deleteModel: (id: string) => request.delete(`/model-hub/models/${id}/`),

  // Approval workflows
  getApprovalRequests: (params?: { status?: string }) =>
    request.get<ApprovalRequest[]>("/model-hub/approvals/", { params }),

  getApprovalRequest: (id: string) =>
    request.get<ApprovalRequest>(`/model-hub/approvals/${id}/`),

  submitApproval: (id: string, action: ApprovalAction, comment?: string) =>
    request.post<any>(`/model-hub/approvals/${id}/review/`, { action, comment }),

  // Promotion
  getPromotionRequests: () =>
    request.get<PromotionRequest[]>("/model-hub/promotions/"),

  requestPromotion: (modelId: string, toStage: Stage) =>
    request.post<any>("/model-hub/promotions/", {
      model_id: modelId,
      to_stage: toStage,
    }),

  approvePromotion: (requestId: string) =>
    request.post<any>(`/model-hub/promotions/${requestId}/approve/`),

  rejectPromotion: (requestId: string) =>
    request.post<any>(`/model-hub/promotions/${requestId}/reject/`),

  // Access control
  getAccessPolicies: (modelId: string) =>
    request.get<AccessPolicy[]>(`/model-hub/models/${modelId}/access/`),

  grantAccess: (
    modelId: string,
    userId: string,
    permission: Permission,
    expiresAt?: string,
  ) =>
    request.post<any>(`/model-hub/models/${modelId}/access/`, {
      user_id: userId,
      permission,
      expires_at: expiresAt,
    }),

  revokeAccess: (policyId: string) =>
    request.delete(`/model-hub/access/${policyId}/`),

  // Catalog search (full‑text search)
  searchCatalog: (params: CatalogSearchParams) =>
    request.get<PaginatedResponse<Model>>("/model-hub/catalog/search/", { params }),
};
