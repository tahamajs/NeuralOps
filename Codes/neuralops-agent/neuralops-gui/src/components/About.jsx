import React, { useState, useEffect } from 'react';
import { Info, Heart, ExternalLink, RefreshCw, Github } from 'lucide-react';

export default function About() {
  const [version, setVersion] = useState('1.0.0');
  const [updateStatus, setUpdateStatus] = useState('checking');

  useEffect(() => {
    if (window.neuralopsAPI) window.neuralopsAPI.getAppVersion().then(v => setVersion(v));
  }, []);

  return (
    <div className="space-y-6 max-w-2xl">
      <h2 className="text-2xl font-bold">About</h2>

      <div className="bg-dark-card border border-dark-border rounded-xl p-8 text-center">
        <div className="w-20 h-20 bg-gradient-to-br from-primary-600 to-purple-600 rounded-2xl flex items-center justify-center mx-auto mb-4">
          <Info size={36} className="text-white" />
        </div>
        <h1 className="text-2xl font-bold">NeuralOps GPU Agent</h1>
        <p className="text-gray-400 mt-1">Version {version}</p>
        <p className="text-gray-500 text-sm mt-4 max-w-md mx-auto">
          Share your idle GPU resources and earn money by renting them out on the NeuralOps decentralized compute marketplace.
        </p>
      </div>

      <div className="bg-dark-card border border-dark-border rounded-xl p-6">
        <h3 className="font-semibold mb-4">System Information</h3>
        <div className="space-y-2 text-sm">
          {[
            ['App Version', version],
            ['Electron', '28.x'],
            ['React', '18.x'],
            ['Node.js', '20.x'],
            ['Platform', navigator.platform],
          ].map(([k, v]) => (
            <div key={k} className="flex justify-between py-1.5 border-b border-dark-border"><span className="text-gray-400">{k}</span><span className="font-mono">{v}</span></div>
          ))}
        </div>
      </div>

      <div className="bg-dark-card border border-dark-border rounded-xl p-6">
        <h3 className="font-semibold mb-4">Links</h3>
        <div className="space-y-2">
          {[
            { label: 'NeuralOps Platform', url: 'https://neuralops.com' },
            { label: 'Documentation', url: 'https://docs.neuralops.com' },
            { label: 'GitHub', url: 'https://github.com/neuralops' },
            { label: 'Support', url: 'mailto:support@neuralops.com' },
          ].map(({ label, url }) => (
            <button key={label} onClick={() => window.neuralopsAPI?.openExternal(url)}
              className="w-full flex items-center justify-between px-3 py-2 hover:bg-dark-border rounded-lg transition-colors text-left">
              <span>{label}</span><ExternalLink size={14} className="text-gray-500" />
            </button>
          ))}
        </div>
      </div>

      <div className="text-center text-gray-500 text-sm py-4">
        <p className="flex items-center justify-center gap-1">Made with <Heart size={14} className="text-red-400 fill-red-400" /> by the NeuralOps team</p>
        <p className="mt-1">© 2026 NeuralOps. All rights reserved.</p>
      </div>
    </div>
  );
}