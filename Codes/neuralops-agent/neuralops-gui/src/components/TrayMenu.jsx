import React, { useState, useEffect } from 'react';
import { Play, Pause, DollarSign, Cpu, Activity, ExternalLink, Power, RefreshCw } from 'lucide-react';
import { api } from '../services/api';

export default function TrayMenu({ onClose }) {
  const [isRunning, setIsRunning] = useState(false);
  const [loading, setLoading] = useState(false);
  const [gpuInfo, setGpuInfo] = useState([]);
  const [balance, setBalance] = useState(null);
  const [stats, setStats] = useState(null);

  const fetchStatus = async () => {
    try {
      if (window.neuralopsAPI) {
        const running = await window.neuralopsAPI.getAgentStatus();
        setIsRunning(running);
        const gpus = await window.neuralopsAPI.getGpuInfo();
        setGpuInfo(gpus);
      }
      const [bal, st] = await Promise.allSettled([
        api.getBalance(),
        api.getMarketStats(),
      ]);
      if (bal.status === 'fulfilled') setBalance(bal.value);
      if (st.status === 'fulfilled') setStats(st.value);
    } catch (e) {
      console.error('Failed to fetch tray status:', e);
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 3000);
    return () => clearInterval(interval);
  }, []);

  const toggleAgent = async () => {
    setLoading(true);
    try {
      if (window.neuralopsAPI) {
        if (isRunning) {
          await window.neuralopsAPI.stopAgent();
          setIsRunning(false);
        } else {
          await window.neuralopsAPI.startAgent();
          setIsRunning(true);
        }
      }
    } catch (e) {
      console.error('Error toggling agent:', e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="w-80 bg-dark-card border border-dark-border rounded-xl shadow-2xl overflow-hidden text-sm text-gray-200 select-none">
      {/* Header */}
      <div className="p-4 border-b border-dark-border flex items-center justify-between bg-dark-bg/60">
        <div className="flex items-center gap-2">
          <div className={`w-2.5 h-2.5 rounded-full ${isRunning ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`} />
          <span className="font-bold text-white text-base">NeuralOps Agent</span>
        </div>
        <span className="text-xs font-mono text-gray-400">
          {isRunning ? 'ONLINE' : 'STOPPED'}
        </span>
      </div>

      {/* Main Stats */}
      <div className="p-4 space-y-3">
        {/* Toggle Button */}
        <button
          onClick={toggleAgent}
          disabled={loading}
          className={`w-full py-2 px-4 rounded-lg font-semibold flex items-center justify-center gap-2 transition-all ${
            isRunning
              ? 'bg-red-600/90 hover:bg-red-600 text-white'
              : 'bg-green-600/90 hover:bg-green-600 text-white'
          } disabled:opacity-50`}
        >
          {loading ? (
            <RefreshCw size={16} className="animate-spin" />
          ) : isRunning ? (
            <>
              <Pause size={16} /> Stop Provider Agent
            </>
          ) : (
            <>
              <Play size={16} /> Start Provider Agent
            </>
          )}
        </button>

        {/* GPU Quick Overview */}
        <div className="bg-dark-bg/80 border border-dark-border rounded-lg p-3 space-y-2">
          <div className="flex items-center justify-between text-xs text-gray-400">
            <span className="flex items-center gap-1"><Cpu size={14} /> GPU Telemetry</span>
            <span>{gpuInfo.length > 0 ? `${gpuInfo[0].tempC}°C` : 'No GPU'}</span>
          </div>
          {gpuInfo.length > 0 ? (
            <div>
              <p className="font-medium text-xs truncate text-white">{gpuInfo[0].name}</p>
              <div className="flex justify-between text-xs mt-1 text-gray-400">
                <span>Load: <strong className="text-gray-200">{gpuInfo[0].utilPct}%</strong></span>
                <span>VRAM: <strong className="text-gray-200">{gpuInfo[0].vramMb} MB</strong></span>
              </div>
            </div>
          ) : (
            <p className="text-xs text-gray-500 italic">No NVIDIA GPU detected</p>
          )}
        </div>

        {/* Quick Earnings & Jobs */}
        <div className="grid grid-cols-2 gap-2">
          <div className="bg-dark-bg/80 border border-dark-border rounded-lg p-2.5">
            <span className="text-xs text-gray-400 flex items-center gap-1">
              <DollarSign size={12} className="text-green-400" /> Balance
            </span>
            <p className="font-bold text-sm text-green-400 mt-1">
              ${(balance?.balance_usd || 0).toFixed(2)}
            </p>
          </div>
          <div className="bg-dark-bg/80 border border-dark-border rounded-lg p-2.5">
            <span className="text-xs text-gray-400 flex items-center gap-1">
              <Activity size={12} className="text-blue-400" /> Active Jobs
            </span>
            <p className="font-bold text-sm text-blue-400 mt-1">
              {stats?.active_jobs || 0}
            </p>
          </div>
        </div>
      </div>

      {/* Footer Actions */}
      <div className="p-3 border-t border-dark-border bg-dark-bg/40 flex justify-between items-center text-xs">
        <button
          onClick={() => {
            if (window.neuralopsAPI) {
              window.neuralopsAPI.minimizeWindow();
            }
            if (onClose) onClose();
          }}
          className="text-gray-400 hover:text-white flex items-center gap-1"
        >
          <ExternalLink size={12} /> Dashboard
        </button>
        <button
          onClick={() => {
            if (window.neuralopsAPI) {
              window.neuralopsAPI.closeWindow();
            }
          }}
          className="text-red-400 hover:text-red-300 flex items-center gap-1"
        >
          <Power size={12} /> Quit App
        </button>
      </div>
    </div>
  );
}
