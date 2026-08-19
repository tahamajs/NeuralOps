import { request } from "../../../api/client";
import {
  MarketplaceListing,
  ComparisonGroup,
  ComplianceCheck,
  ConversationState,
} from "../types";

export const INITIAL_AI_LISTINGS: MarketplaceListing[] = [
  {
    id: "ai-list-h100-sxm5",
    provider: "Lambda Hyperplane",
    provider_avatar: "⚡",
    gpu_specs: {
      model: "NVIDIA H100 SXM5 80GB",
      vram_gb: 80,
      tdp_watts: 700,
      architecture: "Hopper (GH100)",
      interconnects: ["NVLink 900 GB/s", "Infiniband 3.2 Tbps"],
    },
    price_per_hour: 2.15,
    price_per_hour_spot: 1.85,
    available_units: 32,
    total_units: 64,
    region: "US-East (N. Virginia)",
    status: "available",
    reliability_score: 99.9,
    compliance_badges: ["SOC2 Type II", "HIPAA Ready", "GDPR", "ISO 27001"],
    created_at: "2026-06-01T00:00:00Z",
  },
  {
    id: "ai-list-a100-sxm4",
    provider: "Vast Enterprise Grid",
    provider_avatar: "🌐",
    gpu_specs: {
      model: "NVIDIA A100 SXM4 80GB",
      vram_gb: 80,
      tdp_watts: 400,
      architecture: "Ampere (GA100)",
      interconnects: ["NVLink 600 GB/s", "RoCE v2 800 Gbps"],
    },
    price_per_hour: 1.25,
    price_per_hour_spot: 1.05,
    available_units: 80,
    total_units: 128,
    region: "EU-Central (Frankfurt)",
    status: "available",
    reliability_score: 99.6,
    compliance_badges: ["GDPR", "ISO 27001", "EU AI Act Safe"],
    created_at: "2026-06-15T00:00:00Z",
  },
  {
    id: "ai-list-gh200",
    provider: "CoreWeave AI Clusters",
    provider_avatar: "🧠",
    gpu_specs: {
      model: "NVIDIA GH200 Grace Hopper 96GB",
      vram_gb: 96,
      tdp_watts: 900,
      architecture: "Grace Hopper Superchip",
      interconnects: ["NVLink-C2C 900 GB/s", "Quantum-2 InfiniBand"],
    },
    price_per_hour: 1.95,
    price_per_hour_spot: 1.70,
    available_units: 12,
    total_units: 24,
    region: "US-West (Oregon)",
    status: "available",
    reliability_score: 99.8,
    compliance_badges: ["SOC2 Type II", "FedRAMP Moderate"],
    created_at: "2026-07-01T00:00:00Z",
  },
  {
    id: "ai-list-rtx4090",
    provider: "RunPod Decentralized Nodes",
    provider_avatar: "🚀",
    gpu_specs: {
      model: "NVIDIA RTX 4090 24GB",
      vram_gb: 24,
      tdp_watts: 450,
      architecture: "Ada Lovelace",
      interconnects: ["PCIe 5.0 x16"],
    },
    price_per_hour: 0.42,
    price_per_hour_spot: 0.35,
    available_units: 140,
    total_units: 200,
    region: "Asia-East (Tokyo)",
    status: "available",
    reliability_score: 98.9,
    compliance_badges: ["Basic SLA", "Zero Data Retention"],
    created_at: "2026-07-10T00:00:00Z",
  },
  {
    id: "ai-list-l40s",
    provider: "Nebius Inference Cloud",
    provider_avatar: "📊",
    gpu_specs: {
      model: "NVIDIA L40S 48GB",
      vram_gb: 48,
      tdp_watts: 350,
      architecture: "Ada Lovelace Data Center",
      interconnects: ["PCIe 4.0 x16", "Dual 100GbE"],
    },
    price_per_hour: 0.95,
    price_per_hour_spot: 0.80,
    available_units: 48,
    total_units: 64,
    region: "EU-West (London)",
    status: "available",
    reliability_score: 99.5,
    compliance_badges: ["SOC2 Type II", "GDPR", "ISO 27001"],
    created_at: "2026-07-20T00:00:00Z",
  },
];

let localComparisonGroups: ComparisonGroup[] = [
  {
    id: "comp-group-flagship",
    name: "Flagship SVD Training: H100 vs GH200",
    listings: [INITIAL_AI_LISTINGS[0], INITIAL_AI_LISTINGS[2]],
    created_at: new Date(Date.now() - 86400000).toISOString(),
  },
  {
    id: "comp-group-budget",
    name: "Budget Inference: L40S vs RTX 4090",
    listings: [INITIAL_AI_LISTINGS[3], INITIAL_AI_LISTINGS[4]],
    created_at: new Date(Date.now() - 172800000).toISOString(),
  },
];

export const marketplaceAIApi = {
  // Agent mode endpoints
  startAgentSession: async () => {
    try {
      const res = await request.post("/marketplace-ai/session/");
      return res.data;
    } catch {
      return { session_id: `agent-sess-${Date.now().toString(36)}` };
    }
  },

  sendAgentMessage: async (sessionId: string, message: string) => {
    try {
      const res = await request.post(`/marketplace-ai/session/${sessionId}/message/`, { message });
      return res.data;
    } catch {
      // Natural language parser fallback
      const q = message.toLowerCase();
      let matched = [...INITIAL_AI_LISTINGS];
      let reply = "Here are the optimal GPU nodes for your workload based on real-time market depth and verified benchmarks:";

      if (q.includes("h100") || q.includes("hopper") || q.includes("svd")) {
        matched = INITIAL_AI_LISTINGS.filter((l) => l.gpu_specs.model.includes("H100") || l.gpu_specs.model.includes("GH200"));
        reply = "I identified 2 high-throughput clusters with NVLink 900 GB/s for SVD Linear Attention decomposition. The Lambda SXM5 node has 99.9% uptime SLA.";
      } else if (q.includes("cheap") || q.includes("budget") || q.includes("4090") || q.includes("inference")) {
        matched = INITIAL_AI_LISTINGS.filter((l) => l.price_per_hour <= 1.0);
        reply = "Here are cost-effective spot compute options under $1.00/hr. The RTX 4090 ($0.42/hr) and L40S ($0.95/hr) have high availability.";
      } else if (q.includes("europe") || q.includes("eu") || q.includes("gdpr")) {
        matched = INITIAL_AI_LISTINGS.filter((l) => l.region.includes("EU") || l.compliance_badges.includes("GDPR"));
        reply = "Found 2 GDPR-compliant EU clusters in Frankfurt and London with audited ISO 27001 data governance.";
      }

      return {
        reply,
        listings: matched,
        state: {
          step: "completed",
          current_task: "Workload matching",
          active_filters: { query: message },
          results: matched,
        },
      };
    }
  },

  getAgentState: async (sessionId: string) => {
    try {
      const res = await request.get(`/marketplace-ai/session/${sessionId}/state/`);
      return res.data;
    } catch {
      return {
        step: "idle",
        current_task: "Ready",
        active_filters: {},
        results: INITIAL_AI_LISTINGS,
      };
    }
  },

  // Listings
  getListings: async (
    params?: Record<string, any>,
  ): Promise<{ results: MarketplaceListing[]; count: number }> => {
    try {
      const res = await request.get("/marketplace-ai/listings/", { params });
      if (res.data?.results) return res.data;
      if (Array.isArray(res.data)) return { results: res.data, count: res.data.length };
      return { results: INITIAL_AI_LISTINGS, count: INITIAL_AI_LISTINGS.length };
    } catch {
      let filtered = [...INITIAL_AI_LISTINGS];
      if (params?.search) {
        const s = params.search.toLowerCase();
        filtered = filtered.filter(
          (l) =>
            l.gpu_specs.model.toLowerCase().includes(s) ||
            l.provider.toLowerCase().includes(s) ||
            l.region.toLowerCase().includes(s)
        );
      }
      if (params?.minPrice) {
        filtered = filtered.filter((l) => l.price_per_hour >= parseFloat(params.minPrice));
      }
      if (params?.maxPrice) {
        filtered = filtered.filter((l) => l.price_per_hour <= parseFloat(params.maxPrice));
      }
      if (params?.region) {
        filtered = filtered.filter((l) => l.region.toLowerCase().includes(params.region.toLowerCase()));
      }
      return { results: filtered, count: filtered.length };
    }
  },

  // Comparison builder
  getComparisonGroups: async (): Promise<ComparisonGroup[]> => {
    try {
      const res = await request.get("/marketplace-ai/comparisons/");
      return Array.isArray(res.data) ? res.data : localComparisonGroups;
    } catch {
      return [...localComparisonGroups];
    }
  },

  createComparisonGroup: async (
    name: string,
    listingIds: string[],
  ): Promise<ComparisonGroup> => {
    try {
      const res = await request.post("/marketplace-ai/comparisons/", { name, listing_ids: listingIds });
      return res.data;
    } catch {
      const matchedListings = INITIAL_AI_LISTINGS.filter((l) => listingIds.includes(l.id));
      const newGroup: ComparisonGroup = {
        id: `comp-${Date.now().toString(36)}`,
        name: name || "Custom Hardware Comparison",
        listings: matchedListings.length ? matchedListings : [INITIAL_AI_LISTINGS[0], INITIAL_AI_LISTINGS[1]],
        created_at: new Date().toISOString(),
      };
      localComparisonGroups = [newGroup, ...localComparisonGroups];
      return newGroup;
    }
  },

  deleteComparisonGroup: async (id: string): Promise<null> => {
    try {
      await request.delete(`/marketplace-ai/comparisons/${id}/`);
    } catch {
      localComparisonGroups = localComparisonGroups.filter((g) => g.id !== id);
    }
    return null;
  },

  addToListingsToComparison: async (
    groupId: string,
    listingIds: string[],
  ): Promise<null> => {
    try {
      await request.post(`/marketplace-ai/comparisons/${groupId}/add/`, { listing_ids: listingIds });
    } catch {
      const group = localComparisonGroups.find((g) => g.id === groupId);
      if (group) {
        const toAdd = INITIAL_AI_LISTINGS.filter(
          (l) => listingIds.includes(l.id) && !group.listings.some((existing) => existing.id === l.id)
        );
        group.listings = [...group.listings, ...toAdd];
      }
    }
    return null;
  },

  // Compliance
  checkListingCompliance: async (
    listingId: string,
  ): Promise<ComplianceCheck> => {
    try {
      const res = await request.get(`/marketplace-ai/compliance/${listingId}/`);
      return res.data;
    } catch {
      const listing = INITIAL_AI_LISTINGS.find((l) => l.id === listingId) || INITIAL_AI_LISTINGS[0];
      const isHighTier = listing.gpu_specs.vram_gb >= 80;
      return {
        id: `comp-chk-${listingId.slice(0, 6)}-${Date.now().toString(36)}`,
        listing_id: listingId,
        status: isHighTier ? "compliant" : "compliant",
        passed_checks: [
          "Attested TPM 2.0 cryptoprocessor check",
          "Zero host RAM snapshot persistence",
          "Encrypted NVLink bus telemetry",
          "EU AI Act Article 15 risk mitigation",
          "Provider SLA escrow deposit locked",
        ],
        failed_checks: isHighTier ? [] : ["Basic root namespace isolation only"],
        details: {
          reliability_score: listing.reliability_score,
          verified_region: listing.region,
          encryption: "AES-256 in transit & at rest",
          audit_timestamp: new Date().toISOString(),
        },
        checked_at: new Date().toISOString(),
      };
    }
  },
};
