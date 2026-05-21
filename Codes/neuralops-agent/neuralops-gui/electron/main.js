const { app, BrowserWindow, Tray, Menu, ipcMain, shell, dialog, Notification, nativeImage } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const Store = require('electron-store');

const store = new Store({ defaults: { autoStart: false, theme: 'dark', startMinimized: true } });

let mainWindow, tray, agentProcess, isAgentRunning = false;

// ── Auto‑start ──────────────────────────────────────────────────
function setAutoStart(enabled) {
  store.set('autoStart', enabled);
  app.setLoginItemSettings({ openAtLogin: enabled, openAsHidden: store.get('startMinimized') });
}

// ── Spawn Python Agent ──────────────────────────────────────────
function startAgent() {
  if (isAgentRunning) return;
  const agentPath = path.join(__dirname, '..', '..', 'neuralops-agent', 'dist', 'neuralops-agent');
  agentProcess = spawn(agentPath, ['--config', path.join(app.getPath('userData'), 'agent-config.yaml')], {
    stdio: ['pipe', 'pipe', 'pipe'],
  });
  isAgentRunning = true;
  agentProcess.stdout.on('data', (d) => mainWindow?.webContents.send('agent-log', d.toString()));
  agentProcess.stderr.on('data', (d) => mainWindow?.webContents.send('agent-error', d.toString()));
  agentProcess.on('close', (code) => { isAgentRunning = false; mainWindow?.webContents.send('agent-exit', code); });
}

function stopAgent() {
  if (agentProcess) { agentProcess.kill('SIGTERM'); agentProcess = null; isAgentRunning = false; }
}

// ── System Tray with Earnings Popup ─────────────────────────────
function createTray() {
  const icon = nativeImage.createFromPath(path.join(__dirname, '..', 'assets', 'tray-icon.png'));
  tray = new Tray(icon.resize({ width: 16, height: 16 }));

  const updateTrayMenu = () => {
    const contextMenu = Menu.buildFromTemplate([
      { label: `NeuralOps GPU Agent v${app.getVersion()}`, enabled: false },
      { type: 'separator' },
      { label: `Status: ${isAgentRunning ? '🟢 Running' : '🔴 Stopped'}`, enabled: false },
      { label: isAgentRunning ? 'Stop Agent' : 'Start Agent', click: () => {
        isAgentRunning ? stopAgent() : startAgent();
        updateTrayMenu();
      }},
      { type: 'separator' },
      { label: 'Show Dashboard', click: () => { mainWindow.show(); mainWindow.focus(); } },
      { label: 'Quick Stats', enabled: false },
      { type: 'separator' },
      { label: 'Quit', click: () => { app.isQuitting = true; app.quit(); } },
    ]);
    tray.setContextMenu(contextMenu);
  };

  updateTrayMenu();
  tray.setToolTip(`NeuralOps GPU Agent — ${isAgentRunning ? 'Running' : 'Stopped'}`);
  tray.on('double-click', () => { mainWindow.isVisible() ? mainWindow.hide() : mainWindow.show(); });
  return tray;
}

// ── Earnings Notification ───────────────────────────────────────
function showEarningsNotification(amount, jobName) {
  if (!Notification.isSupported()) return;
  const n = new Notification({
    title: '💰 Earnings Received!',
    body: `+$${amount.toFixed(2)} from job "${jobName}"`,
    icon: path.join(__dirname, '..', 'assets', 'tray-icon.png'),
    urgency: 'normal',
  });
  n.on('click', () => { mainWindow.show(); mainWindow.focus(); });
  n.show();
  mainWindow?.webContents.send('earnings-update', { amount, jobName });
}

// ── Window ──────────────────────────────────────────────────────
function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280, height: 860, minWidth: 960, minHeight: 640,
    frame: false, titleBarStyle: 'hidden',
    backgroundColor: '#0f172a',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true, nodeIntegration: false,
    },
    icon: path.join(__dirname, '..', 'assets', 'icon.png'),
    show: !store.get('startMinimized'),
  });

  const isDev = !app.isPackaged;
  isDev ? mainWindow.loadURL('http://localhost:3000') : mainWindow.loadFile(path.join(__dirname, '..', 'dist', 'index.html'));

  mainWindow.on('close', (e) => {
    if (!app.isQuitting) { e.preventDefault(); mainWindow.hide(); }
  });
}

// ── IPC Handlers ────────────────────────────────────────────────
ipcMain.handle('get-agent-status', () => isAgentRunning);
ipcMain.handle('start-agent', () => { startAgent(); return true; });
ipcMain.handle('stop-agent', () => { stopAgent(); return true; });
ipcMain.handle('get-app-version', () => app.getVersion());
ipcMain.handle('open-external', (_, url) => shell.openExternal(url));
ipcMain.handle('get-store', (_, key) => store.get(key));
ipcMain.handle('set-store', (_, key, val) => { store.set(key, val); return true; });
ipcMain.handle('set-auto-start', (_, enabled) => setAutoStart(enabled));
ipcMain.handle('show-earnings-notification', (_, amount, jobName) => showEarningsNotification(amount, jobName));
ipcMain.handle('minimize-window', () => mainWindow.minimize());
ipcMain.handle('maximize-window', () => mainWindow.isMaximized() ? mainWindow.unmaximize() : mainWindow.maximize());
ipcMain.handle('close-window', () => mainWindow.hide());
ipcMain.handle('get-platform', () => process.platform);
ipcMain.handle('show-open-dialog', async (_, opts) => dialog.showOpenDialog(mainWindow, opts));
ipcMain.handle('get-gpu-info', () => {
  try {
    const { execSync } = require('child_process');
    const out = execSync('nvidia-smi --query-gpu=name,memory.total,utilization.gpu,temperature.gpu,fan.speed,power.draw --format=csv,noheader,nounits', { timeout: 5000 }).toString();
    return out.trim().split('\n').map(line => {
      const [name, mem, util, temp, fan, power] = line.split(',').map(s => s.trim());
      return { name, vramMb: parseInt(mem), utilPct: parseInt(util), tempC: parseInt(temp), fanPct: parseInt(fan || '0'), powerW: parseFloat(power || '0') };
    });
  } catch { return []; }
});

// ── Lifecycle ───────────────────────────────────────────────────
app.whenReady().then(() => { createTray(); createWindow(); setAutoStart(store.get('autoStart')); });
app.on('before-quit', () => { app.isQuitting = true; stopAgent(); });
app.on('window-all-closed', () => { if (process.platform !== 'darwin') app.quit(); });
app.on('activate', () => { if (BrowserWindow.getAllWindows().length === 0) createWindow(); else mainWindow?.show(); });


// In electron/main.js, add after the existing IPC handlers:

const http = require('http');
const https = require('https');
const { URL } = require('url');

// ── REST API Proxy (so renderer can call Django without CORS) ──
ipcMain.handle('api-request', async (event, { method, path, body, token }) => {
  return new Promise((resolve, reject) => {
    const serverUrl = store.get('serverUrl') || 'http://localhost:8000';
    const url = new URL(serverUrl + '/api' + path);
    const isHttps = url.protocol === 'https:';
    const transport = isHttps ? https : http;

    const options = {
      method: method || 'GET',
      hostname: url.hostname,
      port: url.port,
      path: url.pathname + url.search,
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: `Token ${token}` } : {}),
      },
    };

    const req = transport.request(options, (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        try { resolve(JSON.parse(data)); }
        catch { resolve(data); }
      });
    });
    req.on('error', (err) => reject(err.message));
    if (body) req.write(JSON.stringify(body));
    req.end();
  });
});

// ── WebSocket Proxy ─────────────────────────────────────────────
let wsClient = null;
ipcMain.handle('ws-connect', (event, { token, serverUrl }) => {
  return new Promise((resolve) => {
    const wsUrl = serverUrl.replace('http', 'ws') + '/ws/provider/?token=' + token;
    wsClient = new WebSocket(wsUrl);
    wsClient.on('open', () => resolve(true));
    wsClient.on('message', (data) => {
      mainWindow?.webContents.send('ws-message', data.toString());
    });
    wsClient.on('close', () => mainWindow?.webContents.send('ws-closed'));
    wsClient.on('error', (err) => mainWindow?.webContents.send('ws-error', err.message));
  });
});

ipcMain.handle('ws-send', (event, data) => {
  if (wsClient && wsClient.readyState === WebSocket.OPEN) {
    wsClient.send(JSON.stringify(data));
    return true;
  }
  return false;
});

ipcMain.handle('ws-disconnect', () => {
  if (wsClient) wsClient.close();
  wsClient = null;
  return true;
});