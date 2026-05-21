// src/services/websocket.js
/**
 * Real‑time WebSocket client for live updates.
 * Connects to Django Channels via the Electron main process.
 */
let ws = null;
let reconnectTimer = null;
let listeners = {};

export const wsClient = {
  connect(token, server) {
    const wsUrl = server.replace('http', 'ws') + '/ws/provider/';
    ws = new WebSocket(wsUrl + '?token=' + token);

    ws.onopen = () => {
      console.log('WebSocket connected');
      this.emit('connected');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        this.emit('message', data);
        if (data.type) this.emit(data.type, data);
      } catch (e) {}
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected, reconnecting in 5s...');
      this.emit('disconnected');
      reconnectTimer = setTimeout(() => this.connect(token, server), 5000);
    };

    ws.onerror = (err) => console.error('WebSocket error', err);
  },

  disconnect() {
    clearTimeout(reconnectTimer);
    if (ws) ws.close();
    ws = null;
  },

  send(data) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(data));
    }
  },

  on(event, callback) {
    if (!listeners[event]) listeners[event] = [];
    listeners[event].push(callback);
  },

  off(event, callback) {
    if (!listeners[event]) return;
    listeners[event] = listeners[event].filter(cb => cb !== callback);
  },

  emit(event, data) {
    (listeners[event] || []).forEach(cb => cb(data));
  },
};