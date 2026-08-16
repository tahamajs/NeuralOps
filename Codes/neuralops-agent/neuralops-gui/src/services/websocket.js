// src/services/websocket.js
/**
 * Real-time WebSocket Client for NeuralOps Provider Agent GUI.
 * Communicates with backend Channels via Electron IPC proxy when available,
 * or direct native WebSocket fallback.
 */

let nativeWs = null;
let reconnectTimer = null;
let listeners = {};
let ipcCleanups = [];

export const wsClient = {
  connect(token, server) {
    const sUrl = server || localStorage.getItem('neuralops_server') || 'http://localhost:8000';
    const sToken = token || localStorage.getItem('neuralops_token') || '';

    // Prefer Electron IPC WebSocket proxy
    if (window.neuralopsAPI?.wsConnect) {
      this.cleanupIpc();
      window.neuralopsAPI.wsConnect(sToken, sUrl).then((connected) => {
        if (connected) {
          this.emit('connected');
        }
      });

      if (window.neuralopsAPI.onWsMessage) {
        const unMsg = window.neuralopsAPI.onWsMessage((raw) => {
          try {
            const data = typeof raw === 'string' ? JSON.parse(raw) : raw;
            this.emit('message', data);
            if (data?.type) this.emit(data.type, data);
          } catch (e) {}
        });
        ipcCleanups.push(unMsg);
      }

      if (window.neuralopsAPI.onWsClosed) {
        const unClosed = window.neuralopsAPI.onWsClosed(() => {
          this.emit('disconnected');
          clearTimeout(reconnectTimer);
          reconnectTimer = setTimeout(() => this.connect(sToken, sUrl), 5000);
        });
        ipcCleanups.push(unClosed);
      }

      if (window.neuralopsAPI.onWsOpen) {
        const unOpen = window.neuralopsAPI.onWsOpen(() => {
          this.emit('connected');
        });
        ipcCleanups.push(unOpen);
      }
      return;
    }

    // Direct Browser WebSocket fallback
    try {
      const wsUrl = sUrl.replace(/^http/, 'ws') + `/ws/provider/?token=${encodeURIComponent(sToken)}`;
      nativeWs = new WebSocket(wsUrl);

      nativeWs.onopen = () => {
        this.emit('connected');
      };

      nativeWs.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.emit('message', data);
          if (data?.type) this.emit(data.type, data);
        } catch (e) {}
      };

      nativeWs.onclose = () => {
        this.emit('disconnected');
        clearTimeout(reconnectTimer);
        reconnectTimer = setTimeout(() => this.connect(sToken, sUrl), 5000);
      };

      nativeWs.onerror = (err) => {
        this.emit('error', err);
      };
    } catch (err) {
      console.warn('Direct WebSocket initialization error:', err);
    }
  },

  cleanupIpc() {
    ipcCleanups.forEach((cleanup) => {
      if (typeof cleanup === 'function') cleanup();
    });
    ipcCleanups = [];
  },

  disconnect() {
    clearTimeout(reconnectTimer);
    this.cleanupIpc();
    if (window.neuralopsAPI?.wsDisconnect) {
      window.neuralopsAPI.wsDisconnect();
    }
    if (nativeWs) {
      try { nativeWs.close(); } catch {}
      nativeWs = null;
    }
  },

  send(data) {
    if (window.neuralopsAPI?.wsSend) {
      window.neuralopsAPI.wsSend(data);
      return;
    }
    if (nativeWs && nativeWs.readyState === 1) { // OPEN
      nativeWs.send(typeof data === 'string' ? data : JSON.stringify(data));
    }
  },

  on(event, callback) {
    if (!listeners[event]) listeners[event] = [];
    listeners[event].push(callback);
  },

  off(event, callback) {
    if (!listeners[event]) return;
    listeners[event] = listeners[event].filter((cb) => cb !== callback);
  },

  emit(event, data) {
    (listeners[event] || []).forEach((cb) => {
      try { cb(data); } catch (e) { console.error(`Error in ws listener for ${event}:`, e); }
    });
  },
};

export default wsClient;