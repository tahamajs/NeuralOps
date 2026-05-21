const { app, BrowserWindow, Tray, Menu, ipcMain, shell } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let mainWindow;
let tray;
let agentProcess;
let isAgentRunning = false;

// Spawn the Python agent
function startAgent() {
  if (isAgentRunning) return;
  const agentPath = path.join(__dirname, '..', '..', 'neuralops-agent', 'dist', 'neuralops-agent');
  agentProcess = spawn(agentPath, ['--config', path.join(app.getPath('userData'), 'agent-config.yaml')]);
  isAgentRunning = true;

  agentProcess.stdout.on('data', (data) => {
    mainWindow?.webContents.send('agent-log', data.toString());
  });
  agentProcess.stderr.on('data', (data) => {
    mainWindow?.webContents.send('agent-error', data.toString());
  });
  agentProcess.on('close', (code) => {
    isAgentRunning = false;
    mainWindow?.webContents.send('agent-exit', code);
  });
}

function stopAgent() {
  if (agentProcess) {
    agentProcess.kill('SIGTERM');
    agentProcess = null;
    isAgentRunning = false;
  }
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    minWidth: 900,
    minHeight: 600,
    frame: false,
    titleBarStyle: 'hidden',
    backgroundColor: '#0f172a',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
    icon: path.join(__dirname, '..', 'assets', 'icon.png'),
  });

  // Load the webpack dev server or the built files
  const isDev = process.env.NODE_ENV !== 'production';
  if (isDev) {
    mainWindow.loadURL('http://localhost:3000');
    mainWindow.webContents.openDevTools({ mode: 'detach' });
  } else {
    mainWindow.loadFile(path.join(__dirname, '..', 'dist', 'index.html'));
  }

  // Tray
  tray = new Tray(path.join(__dirname, '..', 'assets', 'tray-icon.png'));
  const contextMenu = Menu.buildFromTemplate([
    { label: 'Show Window', click: () => mainWindow.show() },
    { label: 'Quit', click: () => { app.isQuitting = true; app.quit(); } },
  ]);
  tray.setToolTip('NeuralOps GPU Agent');
  tray.setContextMenu(contextMenu);
  tray.on('double-click', () => mainWindow.show());
}

// IPC handlers
ipcMain.handle('get-agent-status', async () => isAgentRunning);
ipcMain.handle('start-agent', () => startAgent());
ipcMain.handle('stop-agent', () => stopAgent());
ipcMain.handle('get-app-version', () => app.getVersion());
ipcMain.handle('open-external', (_, url) => shell.openExternal(url));

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', () => {
  stopAgent();
});