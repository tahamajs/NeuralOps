import React, { useState, useEffect } from 'react';
import { Save, RefreshCw, CheckCircle, Shield, Sliders, Server, Moon, Sun } from 'lucide-react';
import { api } from '../services/api';
import { useTheme } from '../context/ThemeContext';

export default function Settings() {
  const { theme, toggleTheme } = useTheme();
  const [savedMessage, setSavedMessage] = useState(null);
  const [saving, setSaving] = useState(false);
  const [config, setConfig] = useState({
    base_price_per_hour: '0.50',
    max_concurrent_jobs: 2,
    storage_gb: 100,
    listing_id: null,
    serverUrl: 'http://localhost:8000',
    apiKey: '',
    autoStart: false,
    autoUpdate: true,
  });

  useEffect(() => {
    const loadSettings = async () => {
      let localServer = 'http://localhost:8000';
      let localKey = '';
      let autoStart = false;

      if (window.neuralopsAPI) {
        const sUrl = await window.neuralopsAPI.getStore('serverUrl');
        if (sUrl) localServer = sUrl;
        const sKey = await window.neuralopsAPI.getStore('apiKey');
        if (sKey) localKey = sKey;
        const sAuto = await window.neuralopsAPI.getStore('autoStart');
        if (sAuto !== undefined) autoStart = Boolean(sAuto);
      } else {
        localServer = localStorage.getItem('neuralops_server') || 'http://localhost:8000';
        localKey = localStorage.getItem('neuralops_token') || '';
      }

      try {
        const data = await api.getMyListings();
        if (Array.isArray(data) && data.length > 0) {
          const l = data[0];
          setConfig((prev) => ({
            ...prev,
            base_price_per_hour: l.base_price_per_hour || '0.50',
            max_concurrent_jobs: l.max_rental_hours || 2,
            storage_gb: l.storage_gb || 100,
            listing_id: l.id,
            serverUrl: localServer,
            apiKey: localKey,
            autoStart,
          }));
          return;
        }
      } catch (e) {}

      setConfig((prev) => ({
        ...prev,
        serverUrl: localServer,
        apiKey: localKey,
        autoStart,
      }));
    };

    loadSettings();
  }, []);

  const handleSave = async (e) => {
    e.preventDefault();
    setSaving(true);
    try {
      // 1. Save to local electron-store & localStorage
      if (window.neuralopsAPI) {
        await window.neuralopsAPI.setStore('serverUrl', config.serverUrl);
        await window.neuralopsAPI.setStore('apiKey', config.apiKey);
        await window.neuralopsAPI.setAutoStart(config.autoStart);
        await window.neuralopsAPI.setStore('basePricePerHour', config.base_price_per_hour);
        await window.neuralopsAPI.setStore('maxConcurrentJobs', config.max_concurrent_jobs);
        await window.neuralopsAPI.setStore('storageGb', config.storage_gb);
      }
      localStorage.setItem('neuralops_server', config.serverUrl);
      if (config.apiKey) {
        localStorage.setItem('neuralops_token', config.apiKey);
      }

      // 2. Save to backend listing if ID is available
      if (config.listing_id) {
        await api.updateListing(config.listing_id, {
          base_price_per_hour: config.base_price_per_hour,
          storage_gb: config.storage_gb,
        });
      }

      setSavedMessage('Configuration updated successfully!');
      setTimeout(() => setSavedMessage(null), 3000);
    } catch (err) {
      alert(`Failed to save settings: ${err.message}`);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="space-y-6 max-w-3xl">
      <div>
        <h2 className="text-2xl font-bold text-white">Agent Preferences & Pricing</h2>
        <p className="text-sm text-gray-400 mt-1">
          Configure marketplace pricing, network endpoints, authentication, and background auto-start.
        </p>
      </div>

      <form onSubmit={handleSave} className="space-y-6">
        {/* Marketplace Pricing & Resources */}
        <div className="bg-dark-card border border-dark-border rounded-2xl p-6 shadow-xl space-y-4">
          <div className="flex items-center gap-2 text-white font-semibold pb-2 border-b border-dark-border">
            <Sliders size={18} className="text-primary-400" />
            <h3>Marketplace Capacity & Rates</h3>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="text-xs font-medium text-gray-400 block mb-1.5">
                Base Hourly Price (USD)
              </label>
              <div className="relative">
                <span className="absolute left-3 top-2.5 text-gray-500 font-bold">$</span>
                <input
                  type="number"
                  step="0.01"
                  min="0.05"
                  value={config.base_price_per_hour}
                  onChange={(e) => setConfig({ ...config, base_price_per_hour: e.target.value })}
                  className="w-full bg-dark-bg border border-dark-border rounded-xl pl-8 pr-4 py-2 text-white text-sm outline-none focus:border-primary-500 font-semibold"
                />
              </div>
            </div>

            <div>
              <label className="text-xs font-medium text-gray-400 block mb-1.5">
                Storage Allocation (GB)
              </label>
              <input
                type="number"
                min="10"
                max="5000"
                value={config.storage_gb}
                onChange={(e) => setConfig({ ...config, storage_gb: parseInt(e.target.value) || 10 })}
                className="w-full bg-dark-bg border border-dark-border rounded-xl px-4 py-2 text-white text-sm outline-none focus:border-primary-500 font-semibold"
              />
            </div>
          </div>

          <div>
            <label className="text-xs font-medium text-gray-400 block mb-1.5">
              Max Concurrent Compute Containers
            </label>
            <input
              type="number"
              min="1"
              max="8"
              value={config.max_concurrent_jobs}
              onChange={(e) =>
                setConfig({ ...config, max_concurrent_jobs: parseInt(e.target.value) || 1 })
              }
              className="w-full bg-dark-bg border border-dark-border rounded-xl px-4 py-2 text-white text-sm outline-none focus:border-primary-500 font-semibold"
            />
          </div>
        </div>

        {/* Server & Authentication */}
        <div className="bg-dark-card border border-dark-border rounded-2xl p-6 shadow-xl space-y-4">
          <div className="flex items-center gap-2 text-white font-semibold pb-2 border-b border-dark-border">
            <Server size={18} className="text-primary-400" />
            <h3>Server Connection & Authentication</h3>
          </div>

          <div>
            <label className="text-xs font-medium text-gray-400 block mb-1.5">
              NeuralOps API Server URL
            </label>
            <input
              type="url"
              value={config.serverUrl}
              onChange={(e) => setConfig({ ...config, serverUrl: e.target.value })}
              className="w-full bg-dark-bg border border-dark-border rounded-xl px-4 py-2 text-white text-sm outline-none focus:border-primary-500 font-mono"
            />
          </div>

          <div>
            <label className="text-xs font-medium text-gray-400 block mb-1.5">
              Provider API Secret Key / Auth Token
            </label>
            <input
              type="password"
              placeholder="Paste token (e.g. nops_live_...)"
              value={config.apiKey}
              onChange={(e) => setConfig({ ...config, apiKey: e.target.value })}
              className="w-full bg-dark-bg border border-dark-border rounded-xl px-4 py-2 text-white text-sm outline-none focus:border-primary-500 font-mono"
            />
          </div>
        </div>

        {/* System & Daemon Settings */}
        <div className="bg-dark-card border border-dark-border rounded-2xl p-6 shadow-xl space-y-4">
          <div className="flex items-center gap-2 text-white font-semibold pb-2 border-b border-dark-border">
            <Shield size={18} className="text-primary-400" />
            <h3>System Startup & Daemon Settings</h3>
          </div>

          <div className="flex items-center justify-between py-2 border-b border-dark-border/60">
            <div>
              <p className="text-sm font-medium text-white">Launch Agent on System Startup</p>
              <p className="text-xs text-gray-400">
                Automatically start the background GPU daemon when logging in
              </p>
            </div>
            <input
              type="checkbox"
              checked={config.autoStart}
              onChange={(e) => setConfig({ ...config, autoStart: e.target.checked })}
              className="w-5 h-5 accent-primary-600 rounded cursor-pointer"
            />
          </div>

          <div className="flex items-center justify-between py-2">
            <div>
              <p className="text-sm font-medium text-white">Interface Color Theme</p>
              <p className="text-xs text-gray-400">Current mode: {theme}</p>
            </div>
            <button
              type="button"
              onClick={toggleTheme}
              className="flex items-center gap-2 px-3 py-1.5 rounded-xl border border-dark-border bg-dark-bg text-gray-300 hover:text-white text-xs font-medium"
            >
              {theme === 'dark' ? <Moon size={14} /> : <Sun size={14} />}
              Toggle {theme === 'dark' ? 'Light' : 'Dark'}
            </button>
          </div>
        </div>

        {/* Save button & feedback */}
        <div className="flex items-center justify-between pt-2">
          {savedMessage ? (
            <div className="flex items-center gap-2 text-emerald-400 text-sm font-medium">
              <CheckCircle size={16} />
              <span>{savedMessage}</span>
            </div>
          ) : (
            <div />
          )}

          <button
            type="submit"
            disabled={saving}
            className="flex items-center gap-2 bg-primary-600 hover:bg-primary-700 text-white px-6 py-2.5 rounded-xl font-semibold text-sm shadow-lg shadow-primary-600/30 transition-all disabled:opacity-50"
          >
            {saving ? <RefreshCw size={16} className="animate-spin" /> : <Save size={16} />}
            Save Configuration
          </button>
        </div>
      </form>
    </div>
  );
}