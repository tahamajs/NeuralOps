const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('neuralopsAPI', {
  // Agent lifecycle
  getAgentStatus: () => ipcRenderer.invoke('get-agent-status'),
  startAgent: () => ipcRenderer.invoke('start-agent'),
  stopAgent: () => ipcRenderer.invoke('stop-agent'),

  // App
  getAppVersion: () => ipcRenderer.invoke('get-app-version'),
  openExternal: (url) => ipcRenderer.invoke('open-external', url),
  minimizeWindow: () => ipcRenderer.invoke('minimize-window'),
  maximizeWindow: () => ipcRenderer.invoke('maximize-window'),
  closeWindow: () => ipcRenderer.invoke('close-window'),
  getPlatform: () => ipcRenderer.invoke('get-platform'),

  // Settings store
  getStore: (key) => ipcRenderer.invoke('get-store', key),
  setStore: (key, val) => ipcRenderer.invoke('set-store', key, val),
  setAutoStart: (enabled) => ipcRenderer.invoke('set-auto-start', enabled),

  // Earnings notification
  showEarningsNotification: (amount, jobName) => ipcRenderer.invoke('show-earnings-notification', amount, jobName),

  // GPU detection
  getGpuInfo: () => ipcRenderer.invoke('get-gpu-info'),

  // Log streaming
  onAgentLog: (cb) => { ipcRenderer.on('agent-log', (_, m) => cb(m)); return () => ipcRenderer.removeAllListeners('agent-log'); },
  onAgentError: (cb) => { ipcRenderer.on('agent-error', (_, m) => cb(m)); return () => ipcRenderer.removeAllListeners('agent-error'); },
  onAgentExit: (cb) => { ipcRenderer.on('agent-exit', (_, c) => cb(c)); return () => ipcRenderer.removeAllListeners('agent-exit'); },
  onEarningsUpdate: (cb) => { ipcRenderer.on('earnings-update', (_, d) => cb(d)); return () => ipcRenderer.removeAllListeners('earnings-update'); },
    // ── REST API ──────────────────────────────────────────────────
  apiRequest: (method, path, body) =>
    ipcRenderer.invoke('api-request', {
      method,
      path,
      body,
      token: localStorage.getItem('neuralops_token'),
    }),

  // ── WebSocket ─────────────────────────────────────────────────
  wsConnect: (token, serverUrl) => ipcRenderer.invoke('ws-connect', { token, serverUrl }),
  wsSend: (data) => ipcRenderer.invoke('ws-send', data),
  wsDisconnect: () => ipcRenderer.invoke('ws-disconnect'),
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

});