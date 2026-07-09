// src/services/api.js
/**
 * NeuralOps REST API Client
 * Fully complete – covers all endpoints needed by the provider desktop app.
 * Uses Electron IPC (main process proxy) when available, falls back to direct fetch.
 */

// ── Auth & config helpers ──────────────────────────────────────────
const getToken = () => localStorage.getItem('neuralops_token');
const getServer = () => localStorage.getItem('neuralops_server') || 'http://localhost:8000';

// ── Core request handler ──────────────────────────────────────────
async function _request(method, path, body = null, isFormData = false) {
  const token = getToken();
  const server = getServer();
  const url = `${server}/api${path}`;

  // Prefer Electron IPC (no CORS issues, uses main process HTTP)
  if (window.neuralopsAPI?.apiRequest) {
    try {
      // For FormData (file uploads), we need to send the body as a Base64 or use a different IPC channel.
      // For simplicity, regular JSON requests go through IPC; file uploads use fetch.
      if (!isFormData) {
        return await window.neuralopsAPI.apiRequest(method, path, body);
      }
    } catch (e) {
      console.warn('IPC apiRequest failed, falling back to fetch:', e);
    }
  }

  // Fallback: direct HTTP fetch (used in dev mode or if IPC unavailable)
  const headers = {};
  if (!isFormData) {
    headers['Content-Type'] = 'application/json';
  }
  if (token) {
    headers['Authorization'] = `Token ${token}`;
  }

  const opts = { method, headers };
  if (body) {
    opts.body = isFormData ? body : JSON.stringify(body);
  }

  const resp = await fetch(url, opts);
  if (!resp.ok) {
    let errMsg = `HTTP ${resp.status}`;
    try {
      const err = await resp.json();
      errMsg = err.detail || err.error || JSON.stringify(err);
    } catch {}
    throw new Error(errMsg);
  }

  // For file downloads, return the blob
  if (path.includes('/download/') || path.includes('/export/')) {
    return resp.blob();
  }
  return resp.json();
}

// ── Public API object ─────────────────────────────────────────────
export const api = {
  // ── Authentication ───────────────────────────────────────────
  login: (username, password) =>
    _request('POST', '/auth/login/', { username, password }),
  register: (username, password, email) =>
    _request('POST', '/auth/register/', { username, password, email }),
  logout: () => _request('POST', '/auth/logout/'),
  refreshToken: () => _request('POST', '/auth/refresh-token/'),
  changePassword: (oldPassword, newPassword) =>
    _request('POST', '/auth/change-password/', { old_password: oldPassword, new_password: newPassword }),
  getUserInfo: () => _request('GET', '/auth/user/'),
  updateProfile: (data) => _request('PATCH', '/auth/profile/', data),

  // ── Profiles (public / extended) ─────────────────────────────
  getMyProfile: () => _request('GET', '/profiles/me/'),
  updateMyProfile: (data) => _request('PATCH', '/profiles/me/', data),
  uploadAvatar: (file) => {
    const formData = new FormData();
    formData.append('avatar', file);
    return _request('POST', '/profiles/me/avatar/', formData, true);
  },
  removeAvatar: () => _request('DELETE', '/profiles/me/avatar/remove/'),
  searchUsers: (query) => _request('GET', `/profiles/search/?q=${encodeURIComponent(query)}`),
  getUserProfile: (userId) => _request('GET', `/profiles/user/${userId}/`),
  getUserProfileByUsername: (username) => _request('GET', `/profiles/username/${encodeURIComponent(username)}/`),

  // ── GPU Marketplace ───────────────────────────────────────────
  // Listings
  getMyListings: () => _request('GET', '/gpu/listings/mine/'),
  getListingDetail: (id) => _request('GET', `/gpu/listings/${id}/`),
  createListing: (data) => _request('POST', '/gpu/listings/', data),
  updateListing: (id, data) => _request('PATCH', `/gpu/listings/${id}/`, data),
  deleteListing: (id) => _request('DELETE', `/gpu/listings/${id}/`),
  toggleListing: (id) => _request('POST', `/gpu/listings/${id}/toggle/`),
  registerAgentListing: (data) => _request('POST', '/gpu/listings/register/', data),  // agent registration
  updateListingStatus: (status) => _request('POST', '/gpu/listings/update-status/', { status }),
  listingHeartbeat: () => _request('POST', '/gpu/listings/heartbeat/'),
  searchListings: (params) => {
    const qs = new URLSearchParams(params).toString();
    return _request('GET', `/gpu/listings/?${qs}`);
  },
  favoriteListing: (id) => _request('POST', `/gpu/listings/${id}/favorite/`),
  unfavoriteListing: (id) => _request('DELETE', `/gpu/listings/${id}/unfavorite/`),
  getMyFavorites: () => _request('GET', '/gpu/favorites/'),

  // Orders
  getOrders: (params) => {
    const qs = new URLSearchParams(params).toString();
    return _request('GET', `/gpu/orders/?${qs}`);
  },
  getMyOrders: () => _request('GET', '/gpu/orders/mine/'),
  createOrder: (data) => _request('POST', '/gpu/orders/', data),
  cancelOrder: (id) => _request('DELETE', `/gpu/orders/${id}/`),
  getOrderDetail: (id) => _request('GET', `/gpu/orders/${id}/`),
  placeBid: (orderId, data) => _request('POST', `/gpu/orders/${orderId}/bid/`, data),
  getOrderBids: (orderId) => _request('GET', `/gpu/orders/${orderId}/bids/`),

  // Jobs
  getJobs: (params) => {
    const qs = new URLSearchParams(params).toString();
    return _request('GET', `/gpu/jobs/?${qs}`);
  },
  getJobDetail: (id) => _request('GET', `/gpu/jobs/${id}/`),
  completeJob: (id, data) => _request('POST', `/gpu/jobs/${id}/complete/`, data),
  failJob: (id, data) => _request('POST', `/gpu/jobs/${id}/fail/`, data),
  disputeJob: (id, reason) => _request('POST', `/gpu/jobs/${id}/dispute/`, { reason }),
  provisionJob: (id) => _request('POST', `/gpu/jobs/${id}/provision/`),
  // Agent reporting
  reportJobStatus: (jobId, status, description) =>
    _request('POST', `/gpu/jobs/${jobId}/status/`, { status, description }),
  reportJobCompleted: (jobId, outputHash, logs) =>
    _request('POST', `/gpu/jobs/${jobId}/complete/`, { output_hash: outputHash, logs }),
  reportJobFailed: (jobId, error, logs) =>
    _request('POST', `/gpu/jobs/${jobId}/fail/`, { error, logs }),
  reportJobLogs: (jobId, logs) =>
    _request('POST', `/gpu/jobs/${jobId}/logs/`, { logs }),

  // Market stats
  getMarketStats: () => _request('GET', '/gpu/stats/'),
  getProviderDashboard: () => _request('GET', '/gpu/dashboard/provider/'),

  // Notifications (marketplace‑related)
  getMarketplaceNotifications: () => _request('GET', '/gpu/notifications/'),

  // ── Escrow (Payments) ────────────────────────────────────────
  getBalance: () => _request('GET', '/escrow/balance/'),
  deposit: (amount, description) =>
    _request('POST', '/escrow/deposit/', { amount_usd: amount, description }),
  withdraw: (amount, method = 'credits') =>
    _request('POST', '/escrow/withdraw/', { amount_usd: amount, method }),
  getTransactions: (params) => {
    const qs = new URLSearchParams(params).toString();
    return _request('GET', `/escrow/transactions/?${qs}`);
  },

  // ── Finance (Invoices, Futures, Usage) ──────────────────────
  getInvoices: () => _request('GET', '/finance/invoices/'),
  getInvoiceDetail: (id) => _request('GET', `/finance/invoices/${id}/`),
  getUsage: () => _request('GET', '/finance/usage/'),
  getMySubscription: () => _request('GET', '/finance/subscription/'),
  changePlan: (planId) => _request('POST', `/finance/subscription/change/${planId}/`),
  getPlans: () => _request('GET', '/finance/plans/'),
  getPaymentMethods: () => _request('GET', '/finance/payment-methods/'),
  addPaymentMethod: (data) => _request('POST', '/finance/payment-methods/', data),
  deletePaymentMethod: (id) => _request('DELETE', `/finance/payment-methods/${id}/`),
  // Futures
  getFutures: () => _request('GET', '/finance/futures/'),
  getFuturesDetail: (id) => _request('GET', `/finance/futures/${id}/`),
  placeFuturesOrder: (contractId, data) =>
    _request('POST', `/finance/futures/${contractId}/order/`, data),
  getMyFuturesOrders: () => _request('GET', '/finance/futures/orders/mine/'),
  getMyFuturesPositions: () => _request('GET', '/finance/futures/positions/mine/'),
  cancelFuturesOrder: (id) => _request('POST', `/finance/futures/orders/${id}/cancel/`),
  // Indices
  getComputeIndices: () => _request('GET', '/finance/indices/'),
  getIndexChart: (id, days = 30) => _request('GET', `/finance/indices/${id}/chart/?days=${days}`),

  // ── Benchmarks ──────────────────────────────────────────────
  getBenchmarkSuites: () => _request('GET', '/benchmarks/suites/'),
  getBenchmarkTests: (suiteId) => _request('GET', `/benchmarks/tests/?suite_id=${suiteId}`),
  startBenchmark: (listingId, suiteId) =>
    _request('POST', `/benchmarks/listings/${listingId}/run/`, { suite_id: suiteId }),
  getBenchmarkRuns: () => _request('GET', '/benchmarks/runs/'),
  getBenchmarkRunDetail: (id) => _request('GET', `/benchmarks/runs/${id}/`),
  getMyScores: () => _request('GET', '/benchmarks/dashboard/'),
  getLeaderboard: (params) => {
    const qs = new URLSearchParams(params).toString();
    return _request('GET', `/benchmarks/leaderboard/?${qs}`);
  },

  // ── Federation (Distributed Training) ───────────────────────
  getMyClusters: () => _request('GET', '/federation/clusters/'),
  createCluster: (data) => _request('POST', '/federation/clusters/', data),
  getClusterDetail: (id) => _request('GET', `/federation/clusters/${id}/`),
  deleteCluster: (id) => _request('DELETE', `/federation/clusters/${id}/`),
  addNodeToCluster: (clusterId, listingId) =>
    _request('POST', `/federation/clusters/${clusterId}/nodes/`, { listing_id: listingId }),
  removeNodeFromCluster: (clusterId, nodeId) =>
    _request('DELETE', `/federation/clusters/${clusterId}/nodes/${nodeId}/`),
  nodeHeartbeat: (clusterId, nodeId) =>
    _request('POST', `/federation/clusters/${clusterId}/nodes/${nodeId}/heartbeat/`),
  getClusterTasks: (clusterId) => _request('GET', `/federation/clusters/${clusterId}/tasks/`),
  createFederationTask: (clusterId, data) =>
    _request('POST', `/federation/clusters/${clusterId}/tasks/`, data),
  launchTask: (taskId) => _request('POST', `/federation/tasks/${taskId}/launch/`),
  cancelTask: (taskId) => _request('POST', `/federation/tasks/${taskId}/cancel/`),
  getTaskResult: (taskId) => _request('GET', `/federation/tasks/${taskId}/result/`),
  providerFederationDashboard: () => _request('GET', '/federation/dashboard/provider/'),

  // ── Reputation ──────────────────────────────────────────────
  getMyReputation: () => _request('GET', '/reputation/dashboard/'),
  getReputationScore: (username) => _request('GET', `/reputation/scores/${username}/`),
  getReputationLeaderboard: () => _request('GET', '/reputation/leaderboard/'),
  stakeTokens: (amount) => _request('POST', '/reputation/stakes/stake/', { amount_usd: amount }),
  getMyStakes: () => _request('GET', '/reputation/stakes/'),
  startUnbonding: (stakeId) => _request('POST', `/reputation/stakes/${stakeId}/unbond/`),
  withdrawStake: (stakeId) => _request('POST', `/reputation/stakes/${stakeId}/withdraw/`),
  getSlashingHistory: () => _request('GET', '/reputation/slashing/history/'),
  getSlashingConditions: () => _request('GET', '/reputation/slashing/conditions/'),
  raiseDispute: (jobId, reason, evidence) =>
    _request('POST', `/reputation/disputes/raise/${jobId}/`, { reason, evidence }),
  getMyDisputes: () => _request('GET', '/reputation/disputes/'),
  getDisputeDetail: (id) => _request('GET', `/reputation/disputes/${id}/`),
  castJuryVote: (disputeId, vote, reasoning) =>
    _request('POST', `/reputation/disputes/${disputeId}/vote/`, { vote, reasoning }),
  getMyBadges: () => _request('GET', '/reputation/badges/mine/'),

  // ── Notes (Research Notebook) ───────────────────────────────
  getNotes: (params) => {
    const qs = new URLSearchParams(params).toString();
    return _request('GET', `/notes/v1/notes/?${qs}`);
  },
  createNote: (data) => _request('POST', '/notes/v1/notes/', data),
  getNoteDetail: (id) => _request('GET', `/notes/v1/notes/${id}/`),
  updateNote: (id, data) => _request('PATCH', `/notes/v1/notes/${id}/`, data),
  deleteNote: (id) => _request('DELETE', `/notes/v1/notes/${id}/`),
  generateRunReport: (runId) => _request('POST', `/notes/v1/notes/generate-report/${runId}/`),
  uploadNoteImage: (id, file) => {
    const formData = new FormData();
    formData.append('image', file);
    return _request('POST', `/notes/v1/notes/${id}/upload-image/`, formData, true);
  },

  // ── Paper (LaTeX) ───────────────────────────────────────────
  getPaperProjects: () => _request('GET', '/paper/projects/'),
  createPaperProject: (data) => _request('POST', '/paper/projects/', data),
  getPaperProject: (id) => _request('GET', `/paper/projects/${id}/`),
  compileProject: (id, data) => _request('POST', `/paper/projects/${id}/compile/`, data),
  downloadPdf: (id) => _request('GET', `/paper/projects/${id}/output/pdf/`),

  // ── Experiments (Training Runs) ─────────────────────────────
  getExperiments: () => _request('GET', '/experiments/'),
  createExperiment: (data) => _request('POST', '/experiments/', data),
  startTraining: (config) => _request('POST', '/runs/start-train/', config),
  startEvaluation: (config) => _request('POST', '/runs/start-evaluate/', config),
  startCompression: (config) => _request('POST', '/runs/start-compress/', config),
  getRunDetail: (id) => _request('GET', `/runs/${id}/`),
  stopRun: (id) => _request('POST', `/runs/${id}/stop/`),
  getRunResults: (id) => _request('GET', `/runs/${id}/results/`),

  // ── System & Health ─────────────────────────────────────────
  healthCheck: () => _request('GET', '/health/'),
  getVersion: () => _request('GET', '/version/'),
  getSystemStatus: () => _request('GET', '/system/status/'),
  getGpuInfo: () => _request('GET', '/system/gpu/'),

  // ── Agent version check ──────────────────────────────────────
  checkAgentUpdate: (currentVersion) =>
    _request('GET', `/gpu/agent/latest/?current=${currentVersion}`),
};

export default api;