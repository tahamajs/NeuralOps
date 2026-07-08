import { request as api } from "../../../api/client";
import {
  AuditLogEntry,
  ComplianceFramework,
  Contract,
  SecurityScan,
  AccessPolicy,
} from "../types";

export const governanceAPI = {
  // Audit Logs
  getAuditLogs: (params?: {
    start_date?: string;
    end_date?: string;
    user_id?: string;
    action?: string;
    resource_type?: string;
    page?: number;
    limit?: number;
  }) =>
    api.get<{ results: AuditLogEntry[]; count: number }>(
      "/governance/audit-logs/",
      { params },
    ),
  exportAuditLogs: (params?: any) =>
    api.post("/governance/audit-logs/export/", params, {
      responseType: "blob",
    }),

  // Compliance
  getComplianceFrameworks: () =>
    api.get<ComplianceFramework[]>("/governance/compliance/frameworks/"),
  getComplianceFramework: (id: string) =>
    api.get<ComplianceFramework>(`/governance/compliance/frameworks/${id}/`),
  runComplianceCheck: (frameworkId: string) =>
    api.post(`/governance/compliance/frameworks/${frameworkId}/check/`),
  generateComplianceReport: (
    frameworkId: string,
    format: "pdf" | "csv" = "pdf",
  ) =>
    api.post(
      `/governance/compliance/frameworks/${frameworkId}/report/`,
      { format },
      { responseType: "blob" },
    ),

  // Contracts
  getContracts: () => api.get<Contract[]>("/governance/contracts/"),
  getContract: (id: string) =>
    api.get<Contract>(`/governance/contracts/${id}/`),
  createContract: (data: Partial<Contract>) =>
    api.post<Contract>("/governance/contracts/", data),
  signContract: (id: string, signatureData: string) =>
    api.post(`/governance/contracts/${id}/sign/`, {
      signature_data: signatureData,
    }),
  declineContract: (id: string, reason?: string) =>
    api.post(`/governance/contracts/${id}/decline/`, { reason }),

  // Security Scans
  getSecurityScans: () =>
    api.get<SecurityScan[]>("/governance/security/scans/"),
  getScan: (id: string) =>
    api.get<SecurityScan>(`/governance/security/scans/${id}/`),
  triggerScan: (targetId: string, targetType: string) =>
    api.post<SecurityScan>("/governance/security/scans/trigger/", {
      target_id: targetId,
      target_type: targetType,
    }),

  // Access Policies
  getAccessPolicies: () =>
    api.get<AccessPolicy[]>("/governance/access/policies/"),
  getAccessPolicy: (id: string) =>
    api.get<AccessPolicy>(`/governance/access/policies/${id}/`),
  createAccessPolicy: (data: Partial<AccessPolicy>) =>
    api.post<AccessPolicy>("/governance/access/policies/", data),
  updateAccessPolicy: (id: string, data: Partial<AccessPolicy>) =>
    api.patch<AccessPolicy>(`/governance/access/policies/${id}/`, data),
  deleteAccessPolicy: (id: string) =>
    api.delete(`/governance/access/policies/${id}/`),
  testAccessPolicy: (policyId: string, context: any) =>
    api.post(`/governance/access/policies/${policyId}/test/`, context),
};
