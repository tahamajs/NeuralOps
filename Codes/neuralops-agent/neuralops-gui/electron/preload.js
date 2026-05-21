const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('neuralopsAPI', {
  getAgentStatus: () => ipcRenderer.invoke('get-agent-status'),
  startAgent: () => ipcRenderer.invoke('start-agent'),
  stopAgent: () => ipcRenderer.invoke('stop-agent'),
  getAppVersion: () => ipcRenderer.invoke('get-app-version'),
  openExternal: (url) => ipcRenderer.invoke('open-external', url),
  onAgentLog: (callback) => ipcRenderer.on('agent-log', (_, msg) => callback(msg)),
  onAgentError: (callback) => ipcRenderer.on('agent-error', (_, msg) => callback(msg)),
  onAgentExit: (callback) => ipcRenderer.on('agent-exit', (_, code) => callback(code)),
});