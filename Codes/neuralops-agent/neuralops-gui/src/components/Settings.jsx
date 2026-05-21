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

  const handleSave = () => {
    // Save to local config file via IPC or API
    alert('Settings saved!');
  };

  return (
    <div className="space-y-6 max-w-2xl">
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