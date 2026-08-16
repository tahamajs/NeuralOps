const { app, BrowserWindow, Tray, Menu, ipcMain, shell, dialog, Notification, nativeImage } = require('electron');
const path = require('path');
const { spawn, execSync } = require('child_process');
const http = require('http');
const https = require('https');
const { URL } = require('url');
const Store = require('electron-store');

// Use ws library if available or global WebSocket
let WebSocketClient = typeof WebSocket !== 'undefined' ? WebSocket : null;
try {
  if (!WebSocketClient) {
    WebSocketClient = require('ws');
  }
} catch (e) {
  console.warn('ws package not available in main process, websocket proxy will use fallback if global exists');
}

// Dockerode for direct container control
let dockerode = null;
let docker = null;
try {
  dockerode = require('dockerode');
  docker = new dockerode();
} catch (e) {
  console.warn('dockerode not available or docker daemon unreachable:', e.message);
}

const store = new Store({
  defaults: {
    autoStart: false,
    theme: 'dark',
    startMinimized: false,
    serverUrl: 'http://localhost:8000',
    apiKey: '',
    basePricePerHour: '0.50',
    maxConcurrentJobs: 2,
    storageGb: 100,
  },
});

let mainWindow = null;
let tray = null;
let agentProcess = null;
let isAgentRunning = false;
let wsClient = null;

// Single instance lock
const gotTheLock = app.requestSingleInstanceLock();
if (!gotTheLock) {
  app.quit();
} else {
  app.on('second-instance', () => {
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.show();
      mainWindow.focus();
    }
  });
}

// ── Auto-start Configuration ─────────────────────────────────────
function setAutoStart(enabled) {
  store.set('autoStart', enabled);
  app.setLoginItemSettings({
    openAtLogin: enabled,
    openAsHidden: Boolean(store.get('startMinimized')),
  });
}

// ── Agent Process Management ─────────────────────────────────────
function resolveAgentCommand() {
  const isPackaged = app.isPackaged;
  const projectRoot = path.join(__dirname, '..', '..');

  if (isPackaged) {
    const binName = process.platform === 'win32' ? 'neuralops-agent.exe' : 'neuralops-agent';
    const bundledPath = path.join(process.resourcesPath, 'bin', binName);
    return { cmd: bundledPath, args: [] };
  }

  // Development paths
  const possibleBinaries = [
    path.join(projectRoot, 'dist', process.platform === 'win32' ? 'neuralops-agent.exe' : 'neuralops-agent'),
    path.join(__dirname, '..', '..', 'dist', 'neuralops-agent'),
    path.join(__dirname, '..', '..', 'neuralops-agent', 'dist', 'neuralops-agent'),
  ];

  const fs = require('fs');
  for (const binPath of possibleBinaries) {
    if (fs.existsSync(binPath)) {
      return { cmd: binPath, args: [] };
    }
  }

  // Fallback to python3 agent runner
  const pythonScript = path.join(projectRoot, 'agent', 'main.py');
  const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';
  return { cmd: pythonCmd, args: [pythonScript] };
}

function startAgent() {
  if (isAgentRunning && agentProcess) return;

  const { cmd, args } = resolveAgentCommand();
  const configPath = path.join(app.getPath('userData'), 'agent-config.yaml');
  const launchArgs = [...args, '--config', configPath];

  try {
    agentProcess = spawn(cmd, launchArgs, {
      stdio: ['pipe', 'pipe', 'pipe'],
      env: { ...process.env, PYTHONUNBUFFERED: '1' },
      cwd: path.join(__dirname, '..', '..'),
    });

    isAgentRunning = true;
    updateTrayMenu();
    mainWindow?.webContents.send('agent-status-changed', true);

    agentProcess.stdout.on('data', (data) => {
      const msg = data.toString();
      mainWindow?.webContents.send('agent-log', msg);
    });

    agentProcess.stderr.on('data', (data) => {
      const msg = data.toString();
      mainWindow?.webContents.send('agent-error', msg);
    });

    agentProcess.on('error', (err) => {
      isAgentRunning = false;
      agentProcess = null;
      updateTrayMenu();
      mainWindow?.webContents.send('agent-error', `Agent process error: ${err.message}`);
      mainWindow?.webContents.send('agent-status-changed', false);
    });

    agentProcess.on('close', (code) => {
      isAgentRunning = false;
      agentProcess = null;
      updateTrayMenu();
      mainWindow?.webContents.send('agent-exit', code);
      mainWindow?.webContents.send('agent-status-changed', false);
    });
  } catch (err) {
    isAgentRunning = false;
    mainWindow?.webContents.send('agent-error', `Failed to launch agent: ${err.message}`);
  }
}

function stopAgent() {
  if (agentProcess) {
    try {
      if (process.platform === 'win32') {
        agentProcess.kill();
      } else {
        agentProcess.kill('SIGTERM');
      }
    } catch (e) {
      console.warn('Error killing agent process:', e);
    }
    agentProcess = null;
    isAgentRunning = false;
    updateTrayMenu();
    mainWindow?.webContents.send('agent-status-changed', false);
  }
}

// ── System Tray ──────────────────────────────────────────────────
function updateTrayMenu() {
  if (!tray) return;

  const contextMenu = Menu.buildFromTemplate([
    { label: `NeuralOps GPU Agent v${app.getVersion()}`, enabled: false },
    { type: 'separator' },
    { label: `Status: ${isAgentRunning ? '🟢 Running' : '🔴 Stopped'}`, enabled: false },
    {
      label: isAgentRunning ? 'Stop Agent' : 'Start Agent',
      click: () => {
        isAgentRunning ? stopAgent() : startAgent();
        updateTrayMenu();
      },
    },
    { type: 'separator' },
    {
      label: 'Open Dashboard',
      click: () => {
        if (mainWindow) {
          mainWindow.show();
          mainWindow.focus();
        }
      },
    },
    { type: 'separator' },
    {
      label: 'Quit',
      click: () => {
        app.isQuitting = true;
        stopAgent();
        app.quit();
      },
    },
  ]);

  tray.setContextMenu(contextMenu);
  tray.setToolTip(`NeuralOps GPU Agent — ${isAgentRunning ? 'Online & Running' : 'Stopped'}`);
}

function createTray() {
  const iconPath = path.join(__dirname, '..', 'assets', 'tray-icon.png');
  let icon = nativeImage.createFromPath(iconPath);
  if (icon.isEmpty()) {
    icon = nativeImage.createEmpty();
  }
  tray = new Tray(icon.resize({ width: 16, height: 16 }));

  updateTrayMenu();
  tray.on('double-click', () => {
    if (mainWindow) {
      mainWindow.isVisible() ? mainWindow.hide() : (mainWindow.show(), mainWindow.focus());
    }
  });
  return tray;
}

// ── Notifications ────────────────────────────────────────────────
function showEarningsNotification(amount, jobName) {
  if (!Notification.isSupported()) return;
  const numAmount = typeof amount === 'number' ? amount : parseFloat(amount || 0);
  const n = new Notification({
    title: '💰 Earnings Received!',
    body: `+$${numAmount.toFixed(2)} from job "${jobName || 'Compute Task'}"`,
    icon: path.join(__dirname, '..', 'assets', 'tray-icon.png'),
    urgency: 'normal',
  });
  n.on('click', () => {
    mainWindow?.show();
    mainWindow?.focus();
  });
  n.show();
  mainWindow?.webContents.send('earnings-update', { amount: numAmount, jobName });
}

// ── Window Creation ──────────────────────────────────────────────
function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 860,
    minWidth: 960,
    minHeight: 640,
    frame: false,
    titleBarStyle: 'hidden',
    backgroundColor: '#0f172a',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
    icon: path.join(__dirname, '..', 'assets', 'icon.png'),
    show: !store.get('startMinimized'),
  });

  const isDev = !app.isPackaged;
  if (isDev) {
    mainWindow.loadURL('http://localhost:3000').catch(() => {
      mainWindow.loadFile(path.join(__dirname, '..', 'dist', 'index.html'));
    });
  } else {
    mainWindow.loadFile(path.join(__dirname, '..', 'dist', 'index.html'));
  }

  mainWindow.on('close', (e) => {
    if (!app.isQuitting) {
      e.preventDefault();
      mainWindow.hide();
    }
  });

  mainWindow.once('ready-to-show', () => {
    if (!store.get('startMinimized')) {
      mainWindow.show();
    }
  });
}

// ── IPC Handlers ─────────────────────────────────────────────────
// Agent Lifecycle
ipcMain.handle('get-agent-status', () => isAgentRunning);
ipcMain.handle('start-agent', () => { startAgent(); return true; });
ipcMain.handle('stop-agent', () => { stopAgent(); return true; });

// App & Window Controls
ipcMain.handle('get-app-version', () => app.getVersion());
ipcMain.handle('open-external', (_, url) => shell.openExternal(url));
ipcMain.handle('get-store', (_, key) => store.get(key));
ipcMain.handle('set-store', (_, key, val) => { store.set(key, val); return true; });
ipcMain.handle('set-auto-start', (_, enabled) => setAutoStart(enabled));
ipcMain.handle('show-earnings-notification', (_, amount, jobName) => showEarningsNotification(amount, jobName));
ipcMain.handle('minimize-window', () => mainWindow?.minimize());
ipcMain.handle('maximize-window', () => {
  if (!mainWindow) return false;
  if (mainWindow.isMaximized()) {
    mainWindow.unmaximize();
    return false;
  } else {
    mainWindow.maximize();
    return true;
  }
});
ipcMain.handle('close-window', () => mainWindow?.hide());
ipcMain.handle('get-platform', () => process.platform);
ipcMain.handle('show-open-dialog', async (_, opts) => dialog.showOpenDialog(mainWindow, opts));

// GPU Detection via nvidia-smi
ipcMain.handle('get-gpu-info', () => {
  try {
    const out = execSync(
      'nvidia-smi --query-gpu=name,memory.total,utilization.gpu,temperature.gpu,fan.speed,power.draw --format=csv,noheader,nounits',
      { timeout: 5000 }
    ).toString();

    return out.trim().split('\n').filter(Boolean).map((line, idx) => {
      const [name, mem, util, temp, fan, power] = line.split(',').map((s) => s.trim());
      return {
        index: idx,
        name: name || `NVIDIA GPU ${idx}`,
        vramMb: parseInt(mem) || 0,
        utilPct: parseInt(util) || 0,
        tempC: parseInt(temp) || 0,
        fanPct: parseInt(fan || '0') || 0,
        powerW: parseFloat(power || '0') || 0,
      };
    });
  } catch {
    return [];
  }
});

// ── REST API Proxy (Eliminates Renderer CORS Restrictions) ────────
ipcMain.handle('api-request', async (event, { method, path: reqPath, body, token }) => {
  return new Promise((resolve, reject) => {
    try {
      const serverUrl = store.get('serverUrl') || 'http://localhost:8000';
      const cleanPath = reqPath.startsWith('/') ? reqPath : `/${reqPath}`;
      const url = new URL(`${serverUrl}/api${cleanPath}`);
      const isHttps = url.protocol === 'https:';
      const transport = isHttps ? https : http;

      const headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      };
      if (token) {
        headers['Authorization'] = `Token ${token}`;
      }

      const postData = body ? JSON.stringify(body) : null;
      if (postData) {
        headers['Content-Length'] = Buffer.byteLength(postData);
      }

      const options = {
        method: (method || 'GET').toUpperCase(),
        hostname: url.hostname,
        port: url.port || (isHttps ? 443 : 80),
        path: url.pathname + url.search,
        headers,
        timeout: 30000,
      };

      const req = transport.request(options, (res) => {
        let data = '';
        res.setEncoding('utf8');
        res.on('data', (chunk) => { data += chunk; });
        res.on('end', () => {
          try {
            const parsed = JSON.parse(data);
            if (res.statusCode >= 400) {
              reject(new Error(parsed.detail || parsed.error || `HTTP ${res.statusCode}`));
            } else {
              resolve(parsed);
            }
          } catch {
            if (res.statusCode >= 400) {
              reject(new Error(`HTTP ${res.statusCode}: ${data}`));
            } else {
              resolve(data);
            }
          }
        });
      });

      req.on('timeout', () => {
        req.destroy();
        reject(new Error('API request timed out (30s)'));
      });

      req.on('error', (err) => reject(err));

      if (postData) {
        req.write(postData);
      }
      req.end();
    } catch (err) {
      reject(err);
    }
  });
});

// ── WebSocket Proxy ──────────────────────────────────────────────
ipcMain.handle('ws-connect', (event, { token, serverUrl }) => {
  return new Promise((resolve) => {
    try {
      if (wsClient) {
        try { wsClient.close(); } catch {}
        wsClient = null;
      }

      if (!WebSocketClient) {
        mainWindow?.webContents.send('ws-error', 'WebSocket client library not available');
        resolve(false);
        return;
      }

      const base = (serverUrl || store.get('serverUrl') || 'http://localhost:8000').replace(/^http/, 'ws');
      const wsUrl = `${base}/ws/provider/?token=${encodeURIComponent(token || '')}`;

      wsClient = new WebSocketClient(wsUrl);

      wsClient.on('open', () => {
        mainWindow?.webContents.send('ws-open');
        resolve(true);
      });

      wsClient.on('message', (data) => {
        mainWindow?.webContents.send('ws-message', data.toString());
      });

      wsClient.on('close', () => {
        mainWindow?.webContents.send('ws-closed');
      });

      wsClient.on('error', (err) => {
        mainWindow?.webContents.send('ws-error', err.message);
      });
    } catch (err) {
      mainWindow?.webContents.send('ws-error', err.message);
      resolve(false);
    }
  });
});

ipcMain.handle('ws-send', (event, data) => {
  if (wsClient && wsClient.readyState === 1) { // OPEN
    const payload = typeof data === 'string' ? data : JSON.stringify(data);
    wsClient.send(payload);
    return true;
  }
  return false;
});

ipcMain.handle('ws-disconnect', () => {
  if (wsClient) {
    try { wsClient.close(); } catch {}
    wsClient = null;
  }
  return true;
});

// ── Docker Container Controls ────────────────────────────────────
ipcMain.handle('container-pause', async (_, containerId) => {
  if (!docker) return { error: 'Docker daemon or dockerode not available' };
  try {
    const c = docker.getContainer(containerId);
    await c.pause();
    return { status: 'paused', containerId };
  } catch (err) {
    return { error: err.message };
  }
});

ipcMain.handle('container-unpause', async (_, containerId) => {
  if (!docker) return { error: 'Docker daemon or dockerode not available' };
  try {
    const c = docker.getContainer(containerId);
    await c.unpause();
    return { status: 'unpaused', containerId };
  } catch (err) {
    return { error: err.message };
  }
});

ipcMain.handle('container-stop', async (_, containerId) => {
  if (!docker) return { error: 'Docker daemon or dockerode not available' };
  try {
    const c = docker.getContainer(containerId);
    await c.stop({ t: 10 });
    return { status: 'stopped', containerId };
  } catch (err) {
    return { error: err.message };
  }
});

ipcMain.handle('container-restart', async (_, containerId) => {
  if (!docker) return { error: 'Docker daemon or dockerode not available' };
  try {
    const c = docker.getContainer(containerId);
    await c.restart({ t: 10 });
    return { status: 'restarted', containerId };
  } catch (err) {
    return { error: err.message };
  }
});

ipcMain.handle('container-stats', async (_, containerId) => {
  if (!docker) return { error: 'Docker daemon or dockerode not available' };
  try {
    const c = docker.getContainer(containerId);
    const stats = await c.stats({ stream: false });
    return stats;
  } catch (err) {
    return { error: err.message };
  }
});

ipcMain.handle('container-logs', async (_, containerId, tail) => {
  if (!docker) return 'Docker daemon or dockerode not available';
  try {
    const c = docker.getContainer(containerId);
    const logs = await c.logs({ stdout: true, stderr: true, tail: tail || 200, follow: false });
    return logs.toString('utf-8');
  } catch (err) {
    return `Error fetching container logs: ${err.message}`;
  }
});

ipcMain.handle('container-list', async () => {
  if (!docker) return [];
  try {
    const containers = await docker.listContainers({ all: true });
    return containers.map((c) => ({
      id: c.Id,
      name: c.Names[0]?.replace(/^\//, ''),
      image: c.Image,
      state: c.State,
      status: c.Status,
      created: c.Created,
    }));
  } catch {
    return [];
  }
});

// ── Application Lifecycle ────────────────────────────────────────
app.whenReady().then(() => {
  createTray();
  createWindow();
  setAutoStart(Boolean(store.get('autoStart')));
});

app.on('before-quit', () => {
  app.isQuitting = true;
  stopAgent();
  if (wsClient) {
    try { wsClient.close(); } catch {}
  }
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  } else {
    mainWindow?.show();
  }
});