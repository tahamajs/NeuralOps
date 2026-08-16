const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('neuralopsAPI', {
  // ── Agent Lifecycle ───────────────────────────────────────────
  getAgentStatus: () => ipcRenderer.invoke('get-agent-status'),
  startAgent: () => ipcRenderer.invoke('start-agent'),
  stopAgent: () => ipcRenderer.invoke('stop-agent'),

  // ── App & Window Controls ─────────────────────────────────────
  getAppVersion: () => ipcRenderer.invoke('get-app-version'),
  openExternal: (url) => ipcRenderer.invoke('open-external', url),
  minimizeWindow: () => ipcRenderer.invoke('minimize-window'),
  maximizeWindow: () => ipcRenderer.invoke('maximize-window'),
  closeWindow: () => ipcRenderer.invoke('close-window'),
  getPlatform: () => ipcRenderer.invoke('get-platform'),
  showOpenDialog: (opts) => ipcRenderer.invoke('show-open-dialog', opts),

  // ── Local Settings Store ──────────────────────────────────────
  getStore: (key) => ipcRenderer.invoke('get-store', key),
  setStore: (key, val) => ipcRenderer.invoke('set-store', key, val),
  setAutoStart: (enabled) => ipcRenderer.invoke('set-auto-start', enabled),

  // ── Desktop Notifications ─────────────────────────────────────
  showEarningsNotification: (amount, jobName) =>
    ipcRenderer.invoke('show-earnings-notification', amount, jobName),

  // ── GPU Hardware Telemetry ────────────────────────────────────
  getGpuInfo: () => ipcRenderer.invoke('get-gpu-info'),

  // ── Log & Lifecycle Event Streaming ───────────────────────────
  onAgentLog: (cb) => {
    const handler = (_, msg) => cb(msg);
    ipcRenderer.on('agent-log', handler);
    return () => ipcRenderer.removeListener('agent-log', handler);
  },
  onAgentError: (cb) => {
    const handler = (_, msg) => cb(msg);
    ipcRenderer.on('agent-error', handler);
    return () => ipcRenderer.removeListener('agent-error', handler);
  },
  onAgentExit: (cb) => {
    const handler = (_, code) => cb(code);
    ipcRenderer.on('agent-exit', handler);
    return () => ipcRenderer.removeListener('agent-exit', handler);
  },
  onAgentStatusChanged: (cb) => {
    const handler = (_, isRunning) => cb(isRunning);
    ipcRenderer.on('agent-status-changed', handler);
    return () => ipcRenderer.removeListener('agent-status-changed', handler);
  },
  onEarningsUpdate: (cb) => {
    const handler = (_, data) => cb(data);
    ipcRenderer.on('earnings-update', handler);
    return () => ipcRenderer.removeListener('earnings-update', handler);
  },

  // ── Docker Container Management ───────────────────────────────
  containerPause: (id) => ipcRenderer.invoke('container-pause', id),
  containerUnpause: (id) => ipcRenderer.invoke('container-unpause', id),
  containerStop: (id) => ipcRenderer.invoke('container-stop', id),
  containerRestart: (id) => ipcRenderer.invoke('container-restart', id),
  containerStats: (id) => ipcRenderer.invoke('container-stats', id),
  containerLogs: (id, tail) => ipcRenderer.invoke('container-logs', id, tail),
  containerList: () => ipcRenderer.invoke('container-list'),

  // ── REST API Proxy ────────────────────────────────────────────
  apiRequest: (method, path, body) =>
    ipcRenderer.invoke('api-request', {
      method,
      path,
      body,
      token: localStorage.getItem('neuralops_token') || '',
    }),

  // ── WebSocket Proxy ───────────────────────────────────────────
  wsConnect: (token, serverUrl) =>
    ipcRenderer.invoke('ws-connect', { token, serverUrl }),
  wsSend: (data) => ipcRenderer.invoke('ws-send', data),
  wsDisconnect: () => ipcRenderer.invoke('ws-disconnect'),
  onWsOpen: (callback) => {
    const handler = () => callback();
    ipcRenderer.on('ws-open', handler);
    return () => ipcRenderer.removeListener('ws-open', handler);
  },
  onWsMessage: (callback) => {
    const handler = (_, data) => callback(data);
    ipcRenderer.on('ws-message', handler);
    return () => ipcRenderer.removeListener('ws-message', handler);
  },
  onWsClosed: (callback) => {
    const handler = () => callback();
    ipcRenderer.on('ws-closed', handler);
    return () => ipcRenderer.removeListener('ws-closed', handler);
  },
  onWsError: (callback) => {
    const handler = (_, err) => callback(err);
    ipcRenderer.on('ws-error', handler);
    return () => ipcRenderer.removeListener('ws-error', handler);
  },
});