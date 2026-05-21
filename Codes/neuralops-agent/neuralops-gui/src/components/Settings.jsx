import React, { useState } from 'react';
import { Settings as SettingsIcon, Save } from 'lucide-react';

export default function Settings() {
  const [config, setConfig] = useState({
    basePricePerHour: 0.50,
    maxConcurrentJobs: 2,
    storageGb: 100,
    autoUpdate: true,
    theme: 'dark',
  });
  const [autoStart, setAutoStart] = useState(false);
  const [theme, setThemeState] = useState('dark');
  const { toggleTheme } = useTheme();

  useEffect(() => {
    if (window.neuralopsAPI) {
      window.neuralopsAPI.getStore('autoStart').then(v => setAutoStart(v));
      window.neuralopsAPI.getStore('theme').then(v => setThemeState(v || 'dark'));
    }
  }, []);

  const handleSave = () => {
    // Save to local config file via IPC or API
    alert('Settings saved!');
  };

  return (
    
    <div className="space-y-6 max-w-2xl">

                {/* Auto‑Start */}
        <div className="flex items-center justify-between">
          <div>
            <span className="text-gray-300">Launch on Startup</span>
            <p className="text-gray-500 text-xs mt-0.5">Agent starts automatically when you log in</p>
          </div>
          <button onClick={() => { const nv = !autoStart; setAutoStart(nv); window.neuralopsAPI?.setAutoStart(nv); }}
            className={`w-12 h-6 rounded-full transition-colors ${autoStart ? 'bg-primary-600' : 'bg-dark-border'}`}>
            <div className={`w-5 h-5 rounded-full bg-white transform transition-transform ${autoStart ? 'translate-x-6' : 'translate-x-1'}`} />
          </button>
        </div>

        {/* Theme */}
        <div className="flex items-center justify-between">
          <div>
            <span className="text-gray-300">Dark Mode</span>
            <p className="text-gray-500 text-xs mt-0.5">Toggle dark/light theme</p>
          </div>
          <button onClick={() => { toggleTheme(); setThemeState(theme === 'dark' ? 'light' : 'dark'); }}
            className={`w-12 h-6 rounded-full transition-colors ${theme === 'dark' ? 'bg-primary-600' : 'bg-dark-border'}`}>
            <div className={`w-5 h-5 rounded-full bg-white transform transition-transform ${theme === 'dark' ? 'translate-x-6' : 'translate-x-1'}`} />
          </button>
        </div>

        {/* Start Minimized */}
        <div className="flex items-center justify-between">
          <div>
            <span className="text-gray-300">Start Minimized</span>
            <p className="text-gray-500 text-xs mt-0.5">Start agent minimized to system tray</p>
          </div>
          <button onClick={() => { const nv = !startMinimized; setStartMinimized(nv); window.neuralopsAPI?.setStore('startMinimized', nv); }}
            className={`w-12 h-6 rounded-full transition-colors ${startMinimized ? 'bg-primary-600' : 'bg-dark-border'}`}>
            <div className={`w-5 h-5 rounded-full bg-white transform transition-transform ${startMinimized ? 'translate-x-6' : 'translate-x-1'}`} />
          </button>
        </div>





      <div className="flex items-center gap-2">
        <SettingsIcon size={24} />
        <h2 className="text-2xl font-bold">Settings</h2>
      </div>
      <div className="bg-dark-card border border-dark-border rounded-xl p-6 space-y-4">
        <div>
          <label className="text-gray-400 text-sm">Base Price per Hour (USD)</label>
          <input type="number" step="0.01" value={config.basePricePerHour}
            onChange={e => setConfig({...config, basePricePerHour: e.target.value})}
            className="w-full mt-1 bg-dark-bg border border-dark-border rounded-lg px-3 py-2 text-white"
          />
        </div>
        <div>
          <label className="text-gray-400 text-sm">Max Concurrent Jobs</label>
          <input type="number" value={config.maxConcurrentJobs}
            onChange={e => setConfig({...config, maxConcurrentJobs: parseInt(e.target.value)})}
            className="w-full mt-1 bg-dark-bg border border-dark-border rounded-lg px-3 py-2 text-white"
          />
        </div>
        <div>
          <label className="text-gray-400 text-sm">Storage Limit (GB)</label>
          <input type="number" value={config.storageGb}
            onChange={e => setConfig({...config, storageGb: parseInt(e.target.value)})}
            className="w-full mt-1 bg-dark-bg border border-dark-border rounded-lg px-3 py-2 text-white"
          />
        </div>
        <div className="flex items-center justify-between">
          <span className="text-gray-400">Auto-Update Agent</span>
          <button onClick={() => setConfig({...config, autoUpdate: !config.autoUpdate})}
            className={`w-12 h-6 rounded-full transition-colors ${config.autoUpdate ? 'bg-primary-600' : 'bg-dark-border'}`}>
            <div className={`w-5 h-5 rounded-full bg-white transform transition-transform ${config.autoUpdate ? 'translate-x-6' : 'translate-x-1'}`} />
          </button>
        </div>
        <button onClick={handleSave}
          className="flex items-center gap-2 bg-primary-600 hover:bg-primary-700 px-4 py-2 rounded-lg font-semibold transition-colors">
          <Save size={18} /> Save Settings
        </button>
      </div>
    </div>
  );
}