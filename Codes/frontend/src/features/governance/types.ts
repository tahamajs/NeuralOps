export type ActionType =
  "create" | "update" | "delete" | "login" | "logout" | "export" | "share";
export type SeverityLevel = "info" | "warning" | "critical";
export type ComplianceStatus =
  "compliant" | "non_compliant" | "partial" | "not_applicable";
export type ScanStatus = "completed" | "in_progress" | "failed";
export type ContractStatus = "draft" | "pending" | "signed" | "expired";
export type PolicyEffect = "allow" | "deny";

export interface AuditLogEntry {
  id: string;
  timestamp: string;
  user_id: string;
  username: string;
  action: ActionType;
  resource_type: string;
  resource_id: string;
  resource_name: string;
  ip_address: string;
  user_agent: string;
  details: Record<string, any>;
  severity: SeverityLevel;
}

export interface ComplianceFramework {
  id: string;
  name: string;
  version: string;
  controls: ComplianceControl[];
  last_assessed: string;
  overall_score: number;
}

export interface ComplianceControl {
  id: string;
  title: string;
  description: string;
  status: ComplianceStatus;
  evidence: string[];
  last_checked: string;
  risk_level: SeverityLevel;
  remediation_steps?: string;
}

export interface Contract {
  id: string;
  title: string;
  parties: string[];
  content: string;
  status: ContractStatus;
  created_at: string;
  signed_at?: string;
  expires_at?: string;
  signers: Signer[];
  audit_trail: AuditLogEntry[];
}

export interface Signer {
  id: string;
  name: string;
  email: string;
  status: "pending" | "signed" | "declined";
  signed_at?: string;
  signature_data?: string;
}

export interface SecurityScan {
  id: string;
  target: string;
  target_type: "model" | "dataset" | "code" | "container";
  status: ScanStatus;
  started_at: string;
  completed_at?: string;
  vulnerabilities: Vulnerability[];
  summary: {
    critical: number;
    high: number;
    medium: number;
    low: number;
  };
}

export interface Vulnerability {
  id: string;
  title: string;
  description: string;
  severity: SeverityLevel;
  cve_id?: string;
  affected_component: string;
  remediation: string;
  fixed_in_version?: string;
  discovered_at: string;
}

export interface AccessPolicy {
  id: string;
  name: string;
  description: string;
  effect: PolicyEffect;
  actions: string[];
  resources: string[];
  conditions: Record<string, any>;
  subjects: string[];
  priority: number;
  enabled: boolean;
  created_at: string;
  updated_at: string;
}
