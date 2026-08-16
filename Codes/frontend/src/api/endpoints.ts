// src/api/endpoints.ts
export const ENDPOINTS = {
  AUTH: {
    LOGIN: "/auth/login/",
    REGISTER: "/auth/register/",
    LOGOUT: "/auth/logout/",
    USER: "/auth/user/",
    PROFILE: "/auth/profile/",
    CHANGE_PASSWORD: "/auth/change-password/",
    REFRESH_TOKEN: "/auth/refresh-token/",
  },

  CONFIG: {
    ATTENTIONS: "/config/attentions/",
    COMPRESSORS: "/config/compressors/",
    TRAINERS: "/config/trainers/",
    METHODS: "/config/methods/",
    MODELS: "/config/models/",
    TASKS: "/config/tasks/",
    PRESETS: "/config/presets/",
    LOSSES: "/config/losses/",
    OPTIMIZERS: "/config/optimizers/",
    SCHEDULERS: "/config/schedulers/",
    CALLBACKS: "/config/callbacks/",
    AUGMENTATIONS: "/config/augmentations/",
    VARIANTS: "/config/variants/",
    HPO: "/config/hpo/",
    DISTRIBUTED: "/config/distributed/",
    TRACKING: "/config/tracking/",
    DEPLOYMENT: "/config/deployment/",
    SEARCH_MODELS: "/config/search-models/",
  },

  LAUNCHER: {
    LAUNCH: "/launch/",
    CONFIG_OPTIONS: "/config/all/",
  },

  AGENT: {
    CHATS: "/chats/",
    CHAT_DETAIL: (sessionId: string) => `/chats/${sessionId}/`,
    MESSAGES: (sessionId: string) => `/chats/${sessionId}/messages/`,
    SEND_MESSAGE: (sessionId: string) => `/chats/${sessionId}/send/`,
    CONVERSATION_SUMMARY: (sessionId: string) => `/chats/${sessionId}/summary/`,
    EXPORT_CONVERSATION: (sessionId: string) => `/chats/${sessionId}/export/`,
  },

  PAPERS: {
    SEARCH: "/papers/search/",
    SAVE: "/papers/save/",
    SAVED_LIST: "/papers/saved/",
    PAPER_DETAIL: (paperId: string) => `/papers/saved/${paperId}/`,
    ANNOTATE: (paperId: string) => `/papers/saved/${paperId}/annotate/`,
    DELETE_PAPER: (paperId: string) => `/papers/saved/${paperId}/delete/`,
    LIBRARY_SEARCH: "/papers/library/",
    CITATION_GRAPH: "/papers/citation-graph/",
    BULK_IMPORT: "/papers/bulk-import/",
    RECOMMENDATIONS_FOR_RUN: (runId: string) =>
      `/papers/recommendations/${runId}/`,
    RECOMMENDATIONS_FROM_LIBRARY: "/papers/recommendations/library/",
    SEARCH_FOR_RUN: (runId: string) => `/papers/search-for-run/${runId}/`,
  },

  TRAINING_ANALYSIS: {
    ANALYZE_RUN: "/training/analyze/",
    RUN_ANALYSIS: (runId: string) => `/training/analysis/${runId}/`,
    SUGGEST_NEXT: "/training/suggest-next/",
  },

  MONITOR: {
    START: "/monitor/start/",
  },

  PROMPTS: {
    TEST: "/prompts/test/",
    LIST: "/prompts/",
    DETAIL: (name: string) => `/prompts/${name}/`,
    HISTORY: (name: string) => `/prompts/${name}/history/`,
  },

  EXPERIMENTS: {
    LIST: "/experiments/",
    DETAIL: (pk: string) => `/experiments/${pk}/`,
    START_TRAIN: "/experiments/start_train/",
    START_EVALUATE: "/experiments/start_evaluate/",
    START_COMPRESS: "/experiments/start_compress/",
  },

  RUNS: {
    DETAIL: (pk: string) => `/runs/${pk}/`,
    STOP: (pk: string) => `/runs/${pk}/stop/`,
    RESULTS: (runId: string) => `/runs/${runId}/results/`,
    SAVE: (runId: string) => `/runs/${runId}/save/`,
    METRICS: (runId: string) => `/runs/${runId}/metrics/`,
    ARTIFACTS: (runId: string) => `/runs/${runId}/artifacts/`,
    HISTORY: (runId: string) => `/runs/${runId}/history/`,
    SNAPSHOTS: (runId: string) => `/runs/${runId}/snapshots/`,
    ANALYSIS: (runId: string) => `/runs/${runId}/analysis/`,
    STATUS: (runId: string) => `/runs/${runId}/status/`,
    DRIFT_REPORT: (runId: string) => `/runs/${runId}/drift/`,
    DRIFT_CHECK: (runId: string) => `/runs/${runId}/drift/check/`,
    FAIRNESS: (runId: string) => `/runs/${runId}/fairness/`,
    ROBUSTNESS: (runId: string) => `/runs/${runId}/robustness/`,
    MODEL_CARD: (runId: string) => `/runs/${runId}/model-card/`,
    PUSH_MODEL_CARD: (runId: string) => `/runs/${runId}/model-card/push/`,
    DATA_LINEAGE: (runId: string) => `/runs/${runId}/lineage/`,
    MODEL_LINEAGE: (versionId: string) => `/models/${versionId}/lineage/`,
  },

  SYSTEM: {
    STATUS: "/system/status/",
    GPU: "/system/gpu/",
    HEALTH: "/health/",
    VERSION: "/version/",
  },

  SEARCH: {
    GLOBAL: "/search/",
  },

  ACTIVITY: {
    FEED: "/activity/",
  },

  JOBS: {
    LIST: "/jobs/",
    DETAIL: (jobId: string) => `/jobs/${jobId}/`,
    PAUSE: (jobId: string) => `/jobs/${jobId}/pause/`,
    RESUME: (jobId: string) => `/jobs/${jobId}/resume/`,
    STOP: (jobId: string) => `/jobs/${jobId}/stop/`,
    RETRY: (jobId: string) => `/jobs/${jobId}/retry/`,
    HISTORY: (jobId: string) => `/jobs/${jobId}/history/`,
    NOTES: (jobId: string) => `/jobs/${jobId}/notes/`,
    CONFIG_HISTORY: (jobId: string) => `/jobs/${jobId}/config/history/`,
    COMPARE_CONFIGS: (jobId: string, versionA: number, versionB: number) =>
      `/jobs/${jobId}/config/compare/${versionA}/${versionB}/`,
    CUSTOM_METRICS: (jobId: string) => `/jobs/${jobId}/metrics/`,
    LOG_CUSTOM_METRIC: (jobId: string) => `/jobs/${jobId}/metrics/log/`,
    ENVIRONMENT: (jobId: string) => `/jobs/${jobId}/environment/`,
    SCHEDULING: (jobId: string) => `/jobs/${jobId}/scheduling/`,
    STATS: "/jobs/stats/",
    BULK_DELETE: "/jobs/bulk-delete/",
    MY_JOBS: "/my-jobs/",
  },

  SCHEDULER: {
    STATUS: "/scheduler/status/",
    METRICS: "/scheduler/metrics/",
    FAIR_SHARE: "/scheduler/fairshare/",
    CONFIG: "/scheduler/config/",
  },

  GROUPS: {
    LIST: "/groups/",
    DETAIL: (groupName: string) => `/groups/${groupName}/`,
  },

  GPU_MARKETPLACE: {
    LISTINGS: "/listings/",
    MY_LISTINGS: "/listings/mine/",
    LISTING_DETAIL: (listingId: string) => `/listings/${listingId}/`,
    TOGGLE_ONLINE: (listingId: string) => `/listings/${listingId}/toggle/`,
    FAVORITE: (listingId: string) => `/listings/${listingId}/favorite/`,
    UNFAVORITE: (listingId: string) => `/listings/${listingId}/unfavorite/`,
    ORDERS: "/orders/",
    MY_ORDERS: "/orders/mine/",
    ORDER_DETAIL: (orderId: string) => `/orders/${orderId}/`,
    ORDER_BIDS: (orderId: string) => `/orders/${orderId}/bids/`,
    PLACE_BID: (orderId: string) => `/orders/${orderId}/bid/`,
    JOBS: "/jobs/",
    JOB_DETAIL: (jobId: string) => `/jobs/${jobId}/`,
    JOB_COMPLETE: (jobId: string) => `/jobs/${jobId}/complete/`,
    JOB_FAIL: (jobId: string) => `/jobs/${jobId}/fail/`,
    JOB_DISPUTE: (jobId: string) => `/jobs/${jobId}/dispute/`,
    JOB_PROVISION: (jobId: string) => `/jobs/${jobId}/provision/`,
    FAVORITES: "/favorites/",
    PRICING_RULES: "/pricing/",
    STATS: "/stats/",
    PROVIDER_DASHBOARD: "/dashboard/provider/",
    REGISTER_LISTING: "/listings/register/",
    AGENT_UPDATE_STATUS: "/listings/update-status/",
    AGENT_HEARTBEAT: "/listings/heartbeat/",
    JOB_UPDATE_STATUS: (jobId: string) => `/jobs/${jobId}/status/`,
    JOB_UPDATE_LOGS: (jobId: string) => `/jobs/${jobId}/logs/`,
    AGENT_CHECK_UPDATE: "/agent/check-update/",
    AGENT_LATEST_VERSION: "/agent/latest/",
  },

  GPU_ESCROW: {
    BALANCE: "/balance/",
    DEPOSIT: "/deposit/",
    WITHDRAW: "/withdraw/",
    TRANSACTIONS: "/transactions/",
  },

  GPU_BENCHMARK: {
    SUITES: "/suites/",
    SUITE_DETAIL: (suiteId: string) => `/suites/${suiteId}/`,
    TESTS: "/tests/",
    RUNS: "/runs/",
    RUN_DETAIL: (runId: string) => `/runs/${runId}/`,
    START_BENCHMARK: (listingId: string) => `/listings/${listingId}/run/`,
    SCORES: "/scores/",
    SCORE_DETAIL: (listingId: string) => `/scores/${listingId}/`,
    RECALCULATE_SCORE: (listingId: string) =>
      `/scores/${listingId}/recalculate/`,
    LEADERBOARD: "/leaderboard/",
    MY_DASHBOARD: "/dashboard/",
  },

  GPU_FEDERATION: {
    CLUSTERS: "/clusters/",
    CLUSTER_DETAIL: (clusterId: string) => `/clusters/${clusterId}/`,
    ADD_NODE: (clusterId: string) => `/clusters/${clusterId}/nodes/`,
    REMOVE_NODE: (clusterId: string, nodeId: string) =>
      `/clusters/${clusterId}/nodes/${nodeId}/`,
    NODE_HEARTBEAT: (clusterId: string, nodeId: string) =>
      `/clusters/${clusterId}/nodes/${nodeId}/heartbeat/`,
    TASKS: (clusterId: string) => `/clusters/${clusterId}/tasks/`,
    TASK_DETAIL: (taskId: string) => `/tasks/${taskId}/`,
    LAUNCH_TASK: (taskId: string) => `/tasks/${taskId}/launch/`,
    TASK_CANCEL: (taskId: string) => `/tasks/${taskId}/cancel/`,
    TASK_RESULT: (taskId: string) => `/tasks/${taskId}/result/`,
    PROVIDER_DASHBOARD: "/dashboard/provider/",
  },

  GPU_REPUTATION: {
    SCORES: (username: string) => `/scores/${username}/`,
    LEADERBOARD: "/leaderboard/",
    MY_STAKES: "/stakes/",
    STAKE_TOKENS: "/stakes/stake/",
    START_UNBONDING: (stakeId: string) => `/stakes/${stakeId}/unbond/`,
    WITHDRAW_STAKE: (stakeId: string) => `/stakes/${stakeId}/withdraw/`,
    SLASHING_HISTORY: "/slashing/history/",
    SLASHING_CONDITIONS: "/slashing/conditions/",
    MY_DISPUTES: "/disputes/",
    RAISE_DISPUTE: (jobId: string) => `/disputes/raise/${jobId}/`,
    DISPUTE_DETAIL: (disputeId: string) => `/disputes/${disputeId}/`,
    SELECT_JURY: (disputeId: string) => `/disputes/${disputeId}/jury/`,
    CAST_JURY_VOTE: (disputeId: string) => `/disputes/${disputeId}/vote/`,
    BADGES: "/badges/",
    MY_BADGES: "/badges/mine/",
    COMPUTE_TRUST_GRAPH: "/trust/compute/",
    TRUST_RANKING: "/trust/ranking/",
    MY_REPUTATION_DASHBOARD: "/dashboard/",
  },

  GPU_VALIDATOR: {
    REGISTER_VALIDATOR: "/validators/register/",
    MY_VALIDATOR_STATUS: "/validators/me/",
    UPDATE_VALIDATOR: "/validators/me/update/",
    VALIDATOR_LIST: "/validators/",
    VALIDATOR_DETAIL: (nodeId: string) => `/validators/${nodeId}/`,
    VERIFICATION_TASKS: "/tasks/",
    VERIFICATION_TASK_DETAIL: (taskId: string) => `/tasks/${taskId}/`,
    SUBMIT_VERIFICATION: (taskId: string) => `/tasks/${taskId}/submit/`,
    FINALIZE_CONSENSUS: (taskId: string) => `/tasks/${taskId}/finalize/`,
    MY_VERIFICATION_TASKS: "/tasks/mine/",
    PROOF_OF_COMPUTE_DETAIL: (jobId: string) => `/proofs/${jobId}/`,
    GENERATE_PROOF: (jobId: string) => `/proofs/${jobId}/generate/`,
  },

  LOAD_BALANCER: {
    GPU_STATUS: "/loadbalancer/gpu/status/",
    GPU_NODES: "/loadbalancer/gpu/nodes/",
    SCHEDULER_STATUS: "/loadbalancer/scheduler/status/",
    SCHEDULER_QUEUE: "/loadbalancer/scheduler/queue/",
    SCHEDULER_HISTORY: "/loadbalancer/scheduler/history/",
    CLUSTER_NODES: "/loadbalancer/cluster/nodes/",
    CLUSTER_GPU_CHECK: "/loadbalancer/cluster/gpu-check/",
  },

  NOTES: {
    LIST: "/notes/v1/notes/",
    DETAIL: (noteId: string) => `/notes/v1/notes/${noteId}/`,
    REACT: (noteId: string) => `/notes/v1/notes/${noteId}/react/`,
    UPLOAD_IMAGE: (noteId: string) => `/notes/v1/notes/${noteId}/upload-image/`,
    FAVORITE: (noteId: string) => `/notes/v1/notes/${noteId}/favorite/`,
    COMMENTS: (noteId: string) => `/notes/v1/notes/${noteId}/comments/`,
    PIN: (noteId: string) => `/notes/v1/notes/${noteId}/pin/`,
    RUN_NOTES: (runId: string) => `/notes/v1/notes/runs/${runId}/`,
    VERSIONS: (noteId: string) => `/notes/v1/notes/${noteId}/versions/`,
    VERSION_DETAIL: (noteId: string, versionNumber: number) =>
      `/notes/v1/notes/${noteId}/versions/${versionNumber}/`,
    MENTIONS: (noteId: string) => `/notes/v1/notes/${noteId}/mentions/`,
    MARK_MENTION_READ: (mentionId: string) =>
      `/notes/v1/mentions/${mentionId}/read/`,
    SHARE: (noteId: string) => `/notes/v1/notes/${noteId}/share/`,
    UNSHARE: (noteId: string) => `/notes/v1/notes/${noteId}/unshare/`,
    SHARED_WITH_ME: "/notes/v1/notes/shared-with-me/",
    SHARE_LINK: (noteId: string) => `/notes/v1/notes/${noteId}/share-link/`,
    LIST_SHARE_LINKS: (noteId: string) =>
      `/notes/v1/notes/${noteId}/share-links/`,
    REVOKE_SHARE_LINK: (noteId: string, linkId: string) =>
      `/notes/v1/notes/${noteId}/share-links/${linkId}/revoke/`,
    PUBLIC_SHARED: (token: string) => `/shared/${token}/`,
    UPLOAD_HF: (noteId: string) => `/notes/v1/notes/${noteId}/upload-hf/`,
    UPLOAD_HF_SPACE: (noteId: string) =>
      `/notes/v1/notes/${noteId}/upload-hf-space/`,
    SYNC_HF: (noteId: string) => `/notes/v1/notes/${noteId}/sync-hf/`,
    SEARCH_PAPER: "/notes/v1/notes/search-paper/",
    ARTIFACTS: (noteId: string) => `/notes/v1/notes/${noteId}/artifacts/`,
    ARTIFACT_DELETE: (noteId: string, artifactId: string) =>
      `/notes/v1/notes/${noteId}/artifacts/${artifactId}/`,
    ARTIFACT_DOWNLOAD: (noteId: string, artifactId: string) =>
      `/notes/v1/notes/${noteId}/artifacts/${artifactId}/download/`,
    PROTOCOLS: "/notes/v1/protocols/",
    PROTOCOL_DETAIL: (protocolId: string) =>
      `/notes/v1/protocols/${protocolId}/`,
    CREATE_NOTE_LINK: "/notes/v1/notes/links/",
    DELETE_NOTE_LINK: (linkId: string) => `/notes/v1/notes/links/${linkId}/`,
    NOTE_LINKS_GRAPH: (noteId: string) =>
      `/notes/v1/notes/${noteId}/links-graph/`,
    REPLIES: (noteId: string) => `/notes/v1/notes/${noteId}/replies/`,
    RESTORE: (noteId: string) => `/notes/v1/notes/${noteId}/restore/`,
    EXPORT_PDF: (noteId: string) => `/notes/v1/notes/${noteId}/export/pdf/`,
    EXPORT_MD: (noteId: string) => `/notes/v1/notes/${noteId}/export/md/`,
    GENERATE_RUN_REPORT: (runId: string) =>
      `/notes/v1/notes/generate-report/${runId}/`,
    TEMPLATES: "/notes/v1/templates/",
    RENDER_TEMPLATE: (templateId: string) =>
      `/notes/v1/templates/${templateId}/render/`,
    FAVORITE_NOTES: "/notes/v1/notes/favorites/",
    FULLTEXT_SEARCH: "/notes/v1/search/",
    NOTIFICATIONS: "/notes/v1/notifications/",
    MARK_ALL_NOTIFICATIONS_READ: "/notes/v1/notifications/read-all/",
    DASHBOARD: "/notes/v1/dashboard/",
    BULK: "/notes/v1/notes/bulk/",
    BULK_TAGS: "/notes/v1/notes/bulk/tags/",
    STATISTICS: "/notes/v1/notes/statistics/",
    RECENT: "/notes/v1/notes/recent/",
    WEBHOOKS: "/notes/v1/webhooks/",
    WEBHOOK_DELETE: (webhookId: string) => `/notes/v1/webhooks/${webhookId}/`,
    WEBHOOK_TEST: (webhookId: string) =>
      `/notes/v1/webhooks/${webhookId}/test/`,
  },

  MODEL_VERSIONING: {
    ALL_VERSIONS: "/versions/",
    RUN_VERSIONS: (runId: string) => `/runs/${runId}/versions/`,
    CLEANUP_VERSIONS: (runId: string) => `/runs/${runId}/versions/cleanup/`,
    SET_PRODUCTION: (runId: string, versionNumber: number) =>
      `/runs/${runId}/versions/${versionNumber}/set_production/`,
    COMPARE_VERSIONS: (runId: string, versionA: number, versionB: number) =>
      `/runs/${runId}/versions/compare/${versionA}/${versionB}/`,
    VERSION_DETAIL: (versionId: string) => `/versions/${versionId}/`,
    PUBLISH_TO_HUB: (versionId: string) => `/versions/${versionId}/publish/`,
    DOWNLOAD_VERSION: (versionId: string) => `/versions/${versionId}/download/`,
    ARCHIVE_VERSION: (versionId: string) => `/versions/${versionId}/archive/`,
    RESTORE_VERSION: (versionId: string) => `/versions/${versionId}/restore/`,
    CREATE_DIFF: (fromVersionId: string, toVersionId: string) =>
      `/versions/diff/${fromVersionId}/${toVersionId}/`,
    RECONSTRUCT_VERSION: (baseVersionId: string, diffId: string) =>
      `/versions/reconstruct/${baseVersionId}/${diffId}/`,
    LINEAGE_GRAPH: (versionId: string) => `/versions/${versionId}/lineage/`,
    COMPRESSION_METHODS: "/compression-methods/",
    MIGRATE_LIFECYCLE: "/versions/migrate-lifecycle/",
  },

  PAPER_EDITOR: {
    PROJECTS: "/projects/",
    PROJECT_DETAIL: (projectId: string) => `/projects/${projectId}/`,
    CLONE_PROJECT: (projectId: string) => `/projects/${projectId}/clone/`,
    FILES: (projectId: string) => `/projects/${projectId}/files/`,
    FILE_DETAIL: (projectId: string, filename: string) =>
      `/projects/${projectId}/files/${filename}/`,
    RENAME_FILE: (projectId: string, filename: string) =>
      `/projects/${projectId}/files/${filename}/rename/`,
    COMPILE: (projectId: string) => `/projects/${projectId}/compile/`,
    OUTPUT_PDF: (projectId: string) => `/projects/${projectId}/output/pdf/`,
    OUTPUT_LOG: (projectId: string) => `/projects/${projectId}/output/log/`,
    PREVIEW_PDF: (projectId: string) => `/projects/${projectId}/preview/pdf/`,
    TEMPLATES: "/templates/",
    USE_TEMPLATE: (templateId: string) => `/templates/${templateId}/use/`,
    EXPORT: (projectId: string) => `/projects/${projectId}/export/`,
    WORD_COUNT: (projectId: string) => `/projects/${projectId}/wordcount/`,
    ADD_COLLABORATOR: (projectId: string) =>
      `/projects/${projectId}/collaborators/add/`,
    REMOVE_COLLABORATOR: (projectId: string) =>
      `/projects/${projectId}/collaborators/remove/`,
    CREATE_SHARE_LINK: (projectId: string) =>
      `/projects/${projectId}/share-links/`,
    LIST_SHARE_LINKS: (projectId: string) =>
      `/projects/${projectId}/share-links/list/`,
    REVOKE_SHARE_LINK: (projectId: string, linkId: string) =>
      `/projects/${projectId}/share-links/${linkId}/revoke/`,
    PUBLIC_SHARED: (token: string) => `/shared/${token}/`,
    PUBLIC_SHARED_PDF: (token: string) => `/shared/${token}/pdf/`,
    GIT_PUSH: (projectId: string) => `/projects/${projectId}/git/push/`,
    GIT_PULL: (projectId: string) => `/projects/${projectId}/git/pull/`,
    COMMENTS: (projectId: string) => `/projects/${projectId}/comments/`,
    COMMENT_DETAIL: (projectId: string, commentId: string) =>
      `/projects/${projectId}/comments/${commentId}/`,
    SNAPSHOTS: (projectId: string) => `/projects/${projectId}/snapshots/`,
    SNAPSHOT_DETAIL: (projectId: string, versionNumber: number) =>
      `/projects/${projectId}/snapshots/${versionNumber}/`,
    RESTORE_SNAPSHOT: (projectId: string, versionNumber: number) =>
      `/projects/${projectId}/snapshots/${versionNumber}/restore/`,
    COMPILE_HISTORY: (projectId: string) => `/projects/${projectId}/compiles/`,
    UPLOAD_HF: (projectId: string) => `/projects/${projectId}/upload-hf/`,
  },

  ORGANIZATIONS: {
    LIST: "/orgs/",
    DETAIL: (orgId: string) => `/orgs/${orgId}/`,
    MEMBERS: (orgId: string) => `/orgs/${orgId}/members/`,
    CHANGE_MEMBER_ROLE: (orgId: string, userId: number) =>
      `/orgs/${orgId}/members/${userId}/role/`,
    REMOVE_MEMBER: (orgId: string, userId: number) =>
      `/orgs/${orgId}/members/${userId}/remove/`,
    LEAVE: (orgId: string) => `/orgs/${orgId}/leave/`,
    INVITE_MEMBER: (orgId: string) => `/orgs/${orgId}/invite/`,
    MY_INVITATIONS: "/invitations/",
    ACCEPT_INVITATION: (invitationId: string) =>
      `/invitations/${invitationId}/accept/`,
    TEAMS: (orgId: string) => `/orgs/${orgId}/teams/`,
    TEAM_DETAIL: (orgId: string, teamId: string) =>
      `/orgs/${orgId}/teams/${teamId}/`,
    ADD_TEAM_MEMBER: (orgId: string, teamId: string) =>
      `/orgs/${orgId}/teams/${teamId}/members/`,
    REMOVE_TEAM_MEMBER: (orgId: string, teamId: string, userId: number) =>
      `/orgs/${orgId}/teams/${teamId}/members/${userId}/`,
    ORG_EXPERIMENTS: (orgId: string) => `/orgs/${orgId}/experiments/`,
    AUDIT_LOG: (orgId: string) => `/orgs/${orgId}/audit-log/`,
    RESOURCE_GROUPS: (orgId: string) => `/orgs/${orgId}/resource-groups/`,
    ROLES: (orgId: string) => `/orgs/${orgId}/roles/`,
    SUBSCRIPTION: (orgId: string) => `/orgs/${orgId}/subscription/`,
    PLANS: "/plans/",
  },

  PORTFOLIOS: {
    PUBLIC: (username: string) => `/user/${username}/`,
    ACTIVITY_GRAPH: (username: string) => `/user/${username}/activity/`,
    EMBED_WIDGET: (username: string) => `/user/${username}/embed/`,
    MY_PORTFOLIO: "/me/",
    ADD_SHOWCASE_ITEM: "/me/items/",
    REMOVE_SHOWCASE_ITEM: (itemId: string) => `/me/items/${itemId}/`,
    TOGGLE_PIN_ITEM: (itemId: string) => `/me/items/${itemId}/pin/`,
    CREATE_WIDGET: "/me/widget/",
    ENDORSE_SKILL: (username: string) => `/user/${username}/endorse/`,
    REMOVE_ENDORSEMENT: (username: string, skill: string) =>
      `/user/${username}/endorse/${skill}/`,
  },

  COMMUNITY: {
    QUESTIONS: "/questions/",
    QUESTION_DETAIL: (questionId: string) => `/questions/${questionId}/`,
    CLOSE_QUESTION: (questionId: string) => `/questions/${questionId}/close/`,
    RELATED_QUESTIONS: (questionId: string) =>
      `/questions/${questionId}/related/`,
    POST_ANSWER: (questionId: string) => `/questions/${questionId}/answer/`,
    ANSWER_DETAIL: (answerId: string) => `/answers/${answerId}/`,
    ACCEPT_ANSWER: (answerId: string) => `/answers/${answerId}/accept/`,
    VOTE_QUESTION: (questionId: string) => `/questions/${questionId}/vote/`,
    VOTE_ANSWER: (answerId: string) => `/answers/${answerId}/vote/`,
    COMMENT_QUESTION: (questionId: string) =>
      `/questions/${questionId}/comment/`,
    COMMENT_ANSWER: (answerId: string) => `/answers/${answerId}/comment/`,
    START_BOUNTY: (questionId: string) => `/questions/${questionId}/bounty/`,
    BOOKMARK_QUESTION: (questionId: string) =>
      `/questions/${questionId}/bookmark/`,
    UNBOOKMARK_QUESTION: (questionId: string) =>
      `/questions/${questionId}/unbookmark/`,
    TAGS: "/tags/",
    TAG_DETAIL: (slug: string) => `/tags/${slug}/`,
    USER_REPUTATION: (username: string) => `/user/${username}/reputation/`,
    USER_ACTIVITY: (username: string) => `/user/${username}/activity/`,
    LEADERBOARD: "/leaderboard/",
    BADGES: "/badges/",
    MY_BADGES: "/badges/mine/",
    PRIVILEGES: "/privileges/",
    MY_TRUST_LEVEL: "/trust-level/",
    DIGEST_SETTINGS: "/digest/",
  },

  LEADERBOARDS: {
    BENCHMARKS: "/benchmarks/",
    BENCHMARK_DETAIL: (benchmarkId: string) => `/benchmarks/${benchmarkId}/`,
    LEADERBOARD: (benchmarkId: string) =>
      `/benchmarks/${benchmarkId}/leaderboard/`,
    MY_RANK: (benchmarkId: string) => `/benchmarks/${benchmarkId}/my-rank/`,
    SUBMISSIONS: (benchmarkId: string) =>
      `/benchmarks/${benchmarkId}/submissions/`,
    SUBMISSION_DETAIL: (benchmarkId: string, submissionId: string) =>
      `/benchmarks/${benchmarkId}/submissions/${submissionId}/`,
    VOTE: (benchmarkId: string, submissionId: string) =>
      `/benchmarks/${benchmarkId}/submissions/${submissionId}/vote/`,
    REMOVE_VOTE: (benchmarkId: string, submissionId: string) =>
      `/benchmarks/${benchmarkId}/submissions/${submissionId}/vote/remove/`,
    FOLLOW_BENCHMARK: (benchmarkId: string) =>
      `/benchmarks/${benchmarkId}/follow/`,
    UNFOLLOW_BENCHMARK: (benchmarkId: string) =>
      `/benchmarks/${benchmarkId}/unfollow/`,
    TASKS: "/tasks/",
    DATASETS: "/datasets/",
    METRICS: "/metrics/",
    GLOBAL_STATS: "/stats/",
  },

  MESSAGING: {
    CHATS: "/chats/",
    START_CHAT: "/chats/start/",
    CHAT_DETAIL: (roomId: string) => `/chats/${roomId}/`,
    SEND_MESSAGE: (roomId: string) => `/chats/${roomId}/send/`,
    MARK_READ: (roomId: string) => `/chats/${roomId}/read/`,
    DELETE_CHAT: (roomId: string) => `/chats/${roomId}/delete/`,
  },

  PROFILES: {
    MY_PROFILE: "/me/",
    UPLOAD_AVATAR: "/me/avatar/",
    REMOVE_AVATAR: "/me/avatar/remove/",
    SEARCH_USERS: "/search/",
    USER_PROFILE_BY_ID: (userId: number) => `/user/${userId}/`,
    USER_PROFILE_BY_USERNAME: (username: string) => `/username/${username}/`,
    SOCIAL_LOGIN_URLS: "/social-login/",
    SOCIAL_CALLBACK: "/social-callback/",
  },

  REVIEWS: {
    REQUESTS: "/requests/",
    REQUEST_DETAIL: (requestId: string) => `/requests/${requestId}/`,
    SUBMIT_FOR_REVIEW: (requestId: string) => `/requests/${requestId}/submit/`,
    OPEN_FOR_REVIEW: (requestId: string) => `/requests/${requestId}/open/`,
    COMPLETE_REVIEW_ROUND: (requestId: string) =>
      `/requests/${requestId}/complete/`,
    REVIEW_SUMMARY: (requestId: string) => `/requests/${requestId}/summary/`,
    WRITE_REVIEW: (requestId: string) => `/requests/${requestId}/review/`,
    MY_REVIEWS_GIVEN: "/my-reviews/",
    ADD_REVIEW_COMMENT: (reviewId: string) => `/reviews/${reviewId}/comment/`,
    RESOLVE_COMMENT: (commentId: string) => `/comments/${commentId}/resolve/`,
    VOTE_REVIEW_HELPFUL: (reviewId: string) => `/reviews/${reviewId}/vote/`,
    TEMPLATES: "/templates/",
    REVIEW_BADGES: "/badges/",
    MY_REVIEWER_BADGES: "/badges/mine/",
    REVIEW_STATS: "/stats/",
  },

  SWEEPS: {
    LIST: "/sweeps/",
    DETAIL: (sweepId: string) => `/sweeps/${sweepId}/`,
    TRIALS: (sweepId: string) => `/sweeps/${sweepId}/trials/`,
    TRIAL_DETAIL: (sweepId: string, trialNumber: number) =>
      `/sweeps/${sweepId}/trials/${trialNumber}/`,
    TRIAL_UPDATE: (sweepId: string, trialNumber: number) =>
      `/sweeps/${sweepId}/trials/${trialNumber}/update/`,
    REPORT_INTERMEDIATE: (sweepId: string, trialNumber: number) =>
      `/sweeps/${sweepId}/trials/${trialNumber}/intermediate/`,
    SUBMIT_PREFERENCE: (sweepId: string) => `/sweeps/${sweepId}/preference/`,
    SWEEP_IMPORTANCE: (sweepId: string) => `/sweeps/${sweepId}/importance/`,
    SWEEP_PARETO: (sweepId: string) => `/sweeps/${sweepId}/pareto/`,
    SWEEP_CONTOUR: (sweepId: string) => `/sweeps/${sweepId}/contour/`,
    SWEEP_HISTORY: (sweepId: string) => `/sweeps/${sweepId}/history/`,
    SWEEP_SLICE: (sweepId: string) => `/sweeps/${sweepId}/slice/`,
    SWEEP_PARALLEL: (sweepId: string) => `/sweeps/${sweepId}/parallel/`,
    SWEEP_EDF: (sweepId: string) => `/sweeps/${sweepId}/edf/`,
    SWEEP_WARM_START: (sweepId: string) => `/sweeps/${sweepId}/warm-start/`,
    SWEEP_DATAFRAME: (sweepId: string) => `/sweeps/${sweepId}/dataframe/`,
    SWEEP_EXPORT: (sweepId: string) => `/sweeps/${sweepId}/export/`,
    SWEEP_IMPORT: "/sweeps/import/",
  },

  TERMINAL: {
    SESSIONS: "/sessions/",
    KILL_SESSION: (sessionId: string) => `/sessions/${sessionId}/kill/`,
  },

  DASHBOARD: {
    INDEX: "/",
    DASHBOARD_VIEW: "/dashboard/",
    JOB_DETAIL_VIEW: (jobId: string) => `/dashboard/job/${jobId}/`,
    WIDGETS: "/widgets/",
    WIDGET_DATA: (widgetId: string) => `/widgets/${widgetId}/data/`,
    WIDGET_CONFIG: (widgetId: string) => `/widgets/${widgetId}/config/`,
    STATS: "/dashboard/stats/",
    SETTINGS: "/dashboard/settings/",
    MARKETPLACE_LISTINGS: "/marketplace/listings/",
    CONSUMER_ORDERS: "/marketplace/orders/",
    CONSUMER_JOBS: "/marketplace/jobs/",
    CONSUMER_JOB_DETAIL: (jobId: string) => `/marketplace/jobs/${jobId}/`,
    CONSUMER_BALANCE: "/marketplace/balance/",
    CONSUMER_INVOICES: "/marketplace/invoices/",
    CONSUMER_USAGE: "/marketplace/usage/",
    CONSUMER_LEADERBOARD: "/marketplace/leaderboard/",
    CONSUMER_REPUTATION: "/marketplace/reputation/",
  },

  // -----------------------------------------------------
  // ADDED MISSING ENDPOINTS FOR FULL PLATFORM COMPLETION
  // -----------------------------------------------------
  EXPERIMENT_EXPLORER: {
    ROOT: "/explorer/",
  },
  LIVE_DASH: {
    ROOT: "/live-dash/",
  },
  MODEL_HUB: {
    ROOT: "/model-hub/",
  },
  LAB_ASSISTANT: {
    ROOT: "/lab/",
  },
  HEALTH_MONITOR: {
    ROOT: "/health-monitor/",
    HEALTH: "/health-monitor/health/",
    STORES: "/health-monitor/stores/",
    GROUPS: "/health-monitor/groups/",
    MONITORS: "/health-monitor/monitors/",
    MONITOR_CHECK: (monitorId: string) =>
      `/health-monitor/monitors/${monitorId}/check/`,
    MONITOR_HISTORY: (monitorId: string) =>
      `/health-monitor/monitors/${monitorId}/history/`,
    ALERTS: "/health-monitor/alerts/",
    ALERT_ACKNOWLEDGE: (alertId: string) =>
      `/health-monitor/alerts/${alertId}/acknowledge/`,
    CHANNELS: "/health-monitor/channels/",
    CHANNEL_DELETE: (channelId: string) =>
      `/health-monitor/channels/${channelId}/`,
    PERFORMANCE: "/health-monitor/performance/",
  },
  COMMUNITY_LAB: {
    ROOT: "/community-lab/",
  },
  PLAYGROUND: {
    ROOT: "/playground/",
  },
  DATASETS: {
    ROOT: "/datasets/",
    LIST: "/datasets/",
    DETAIL: (id: string) => `/datasets/${id}/`,
    EXPORT: (id: string) => `/datasets/${id}/export/`,
    ARTIFACTS: "/datasets/artifacts/",
    ARTIFACT_LINEAGE: (id: string) => `/datasets/artifacts/${id}/lineage/`,
    AUDIT_LOGS: "/datasets/audit-logs/",
    AUDIT_EXPORT: "/datasets/audit-logs/export/",
  },
  LLM_HUB: {
    ROOT: "/llm/",
    LIST: "/llm/",
  },
  COLLABORATORS: {
    ROOT: "/collaborators/",
  },
  IMPACT_PREDICTOR: {
    ROOT: "/impact/",
  },
  LITERATURE: {
    ROOT: "/literature/",
  },
  LITERATURE_DISCOVERY: {
    ROOT: "/discovery/",
  },
  COSCIENTIST: {
    ROOT: "/coscientist/",
  },
  IDEA_TOURNAMENT: {
    ROOT: "/ideas/",
  },
  COMPUTATIONAL_DISCOVERY: {
    ROOT: "/computational/",
  },
  ACCELERATOR: {
    ROOT: "/accelerator/",
  },
  ANOMALY_HPO: {
    ROOT: "/anomaly-hpo/",
  },
  AUTONOMOUS_RESEARCH: {
    ROOT: "/autonomous-research/",
  },
  MARKETPLACE_AI: {
    ROOT: "/marketplace-ai/",
  },
  CODING_ASSISTANT: {
    ROOT: "/code/",
  },
  EXPERIMENT_RECOMMENDER: {
    ROOT: "/recommender/",
  },
  FORENSICS: {
    ROOT: "/forensics/",
  },
  RESEARCH_COACH: {
    ROOT: "/coach/",
  },
  RESEARCH_AGENT: {
    ROOT: "/agent/",
  },
  RESEARCH: {
    ROOT: "/research/",
  },
  RESEARCH_DISCOVERY: {
    ROOT: "/research-discovery/",
  },
  GOVERNANCE: {
    ROOT: "/governance/",
  },
  INTEGRATIONS: {
    ROOT: "/integrations/",
  },
  MLOPS: {
    ROOT: "/mlops/",
  },
  ATTENTION_DASHBOARD: {
    CANVAS: "/attention-visualization/canvas/",
    INTERPRETABILITY: "/attention-visualization/interpretability/",
    COMPARE: "/attention-visualization/compare/",
  },
  STUDIOS: {
    REPRESENTATION: "/attention-visualization/studios/representation/",
    COMPUTATION: "/attention-visualization/studios/computation/",
    OPTIMIZATION: "/attention-visualization/studios/optimization/",
  },
};
