import React, { useEffect, useState, useRef } from 'react';
import { Cpu, HardDrive, Thermometer, Fan, Zap, Activity, Pause, Play, Square, RefreshCw, Terminal } from 'lucide-react';
import { wsClient } from '../services/websocket';
import { api } from '../services/api';

const GAUGE_COLORS = {
  green: { stroke: '#22c55e', bg: 'rgba(34,197,94,0.1)' },
  yellow: { stroke: '#f59e0b', bg: 'rgba(245,158,11,0.1)' },
  red: { stroke: '#ef4444', bg: 'rgba(239,68,68,0.1)' },
};

function gaugeColor(pct) {
  if (pct > 80) return GAUGE_COLORS.red;
  if (pct > 50) return GAUGE_COLORS.yellow;
  return GAUGE_COLORS.green;
}

function MiniGauge({ value, max, label, icon: Icon }) {
  const pct = Math.min(100, Math.max(0, (value / (max || 1)) * 100));
  const colors = gaugeColor(pct);
  const radius = 28, circumference = 2 * Math.PI * radius;
  const offset = circumference - (pct / 100) * circumference;

  return (
    <div className="flex flex-col items-center">
      <div className="relative w-20 h-20">
        <svg width="80" height="80" viewBox="0 0 80 80">
          <circle cx="40" cy="40" r={radius} fill="none" stroke="#1e293b" strokeWidth="6" />
          <circle cx="40" cy="40" r={radius} fill="none" stroke={colors.stroke} strokeWidth="6"
            strokeDasharray={circumference} strokeDashoffset={offset} strokeLinecap="round"
            transform="rotate(-90 40 40)" style={{ transition: 'stroke-dashoffset 0.5s ease' }} />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-sm font-bold">{Math.round(pct)}%</span>
        </div>
      </div>
      <div className="flex items-center gap-1 mt-1 text-xs text-gray-400">
        <Icon size={12} /> {label}
      </div>
    </div>
  );
}

export default function LiveGPUMonitor() {
  const [gpus, setGpus] = useState([]);
  const [processes, setProcesses] = useState([]);
  const [activeJobs, setActiveJobs] = useState([]);
  const [selectedJob, setSelectedJob] = useState(null);
  const [containerLogs, setContainerLogs] = useState('');
  const [showTerminal, setShowTerminal] = useState(false);
  const [actionFeedback, setActionFeedback] = useState(null);
  const logsEndRef = useRef(null);

  // ── Listen to WebSocket GPU snapshots ─────────────────────────
  useEffect(() => {
    const h = (data) => {
      if (data.type === 'gpu_snapshot') {
        setGpus(data.data?.gpus || []);
        setProcesses(data.data?.processes || []);
      }
    };
    wsClient.on('gpu_snapshot', h);
    wsClient.on('message', h);
    return () => { wsClient.off('gpu_snapshot', h); wsClient.off('message', h); };
  }, []);

  // ── Fetch active jobs ─────────────────────────────────────────
  useEffect(() => {
    const fetchJobs = () => {
      api.getJobs({ status: 'running' }).then(d => setActiveJobs(d.results || [])).catch(() => {});
    };
    fetchJobs();
    const t = setInterval(fetchJobs, 5000);
    return () => clearInterval(t);
  }, []);

  // ── Container control handlers ────────────────────────────────
  const handlePause = async (job) => {
    if (window.neuralopsAPI) {
      const r = await window.neuralopsAPI.containerPause(job.container_id);
      showFeedback(r.status);
    }
  };
  const handleUnpause = async (job) => {
    if (window.neuralopsAPI) {
      const r = await window.neuralopsAPI.containerUnpause(job.container_id);
      showFeedback(r.status);
    }
  };
  const handleStop = async (job) => {
    if (window.neuralopsAPI) {
      const r = await window.neuralopsAPI.containerStop(job.container_id);
      showFeedback(r.status);
    }
  };
  const handleRestart = async (job) => {
    if (window.neuralopsAPI) {
      const r = await window.neuralopsAPI.containerRestart(job.container_id);
      showFeedback(r.status);
    }
  };
  const handleViewLogs = async (job) => {
    setSelectedJob(job);
    setShowTerminal(true);
    if (window.neuralopsAPI) {
      const logs = await window.neuralopsAPI.containerLogs(job.container_id, 500);
      setContainerLogs(logs);
    }
  };

  const showFeedback = (msg) => {
    setActionFeedback(msg);
    setTimeout(() => setActionFeedback(null), 2000);
  };

  // ── Auto‑scroll logs ─────────────────────────────────────────
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [containerLogs]);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Live GPU Monitor</h2>
        {actionFeedback && (
          <span className="bg-green-600/20 text-green-400 px-3 py-1 rounded-full text-sm animate-pulse">
            {actionFeedback}
          </span>
        )}
      </div>

      {/* ── GPU Cards ─────────────────────────────────────────── */}
      {gpus.length === 0 && (
        <div className="bg-dark-card border border-dark-border rounded-xl p-8 text-center">
          <Cpu size={48} className="mx-auto text-gray-600 mb-3" />
          <p className="text-gray-400">No GPU data yet — waiting for agent...</p>
        </div>
      )}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {gpus.map(gpu => (
          <div key={gpu.index} className="bg-dark-card border border-dark-border rounded-xl p-5">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="font-bold text-lg">{gpu.name || `GPU ${gpu.index}`}</h3>
                <p className="text-gray-400 text-xs">
                  {gpu.vram_used_mb}/{gpu.vram_total_mb} MB · {gpu.process_count || processes.filter(p => p.gpu_index === gpu.index).length} processes
                </p>
              </div>
              <div className={`px-2 py-1 rounded-full text-xs font-medium ${gpu.gpu_util_pct > 80 ? 'bg-red-400/10 text-red-400' : gpu.gpu_util_pct > 50 ? 'bg-orange-400/10 text-orange-400' : 'bg-green-400/10 text-green-400'}`}>
                {gpu.gpu_util_pct > 80 ? 'High' : gpu.gpu_util_pct > 50 ? 'Moderate' : 'Idle'}
              </div>
            </div>

            <div className="flex justify-between">
              <MiniGauge value={gpu.gpu_util_pct} max={100} label="GPU" icon={Activity} />
              <MiniGauge value={gpu.vram_pct} max={100} label="VRAM" icon={HardDrive} />
              <MiniGauge value={gpu.temperature_c} max={90} label={`${gpu.temperature_c}°C`} icon={Thermometer} />
              <MiniGauge value={gpu.fan_speed_pct} max={100} label="Fan" icon={Fan} />
            </div>

            <div className="mt-3 grid grid-cols-3 gap-2 text-xs text-gray-400">
              <div>Power: <span className="text-white">{gpu.power_watts}W</span></div>
              <div>SM: <span className="text-white">{gpu.sm_clock_mhz} MHz</span></div>
              <div>Mem: <span className="text-white">{gpu.mem_clock_mhz} MHz</span></div>
            </div>
          </div>
        ))}
      </div>

      {/* ── Active Jobs / Containers ───────────────────────────── */}
      <div className="bg-dark-card border border-dark-border rounded-xl p-5">
        <h3 className="font-semibold mb-4">Active Jobs</h3>
        {activeJobs.length === 0 && <p className="text-gray-500 text-sm">No active jobs</p>}
        <div className="space-y-2">
          {activeJobs.map(job => (
            <div key={job.id} className="flex items-center justify-between py-2 px-3 bg-dark-bg rounded-lg">
              <div>
                <p className="text-sm font-medium truncate max-w-[180px]">{job.listing_gpu_model || job.id.slice(0,8)}</p>
                <p className="text-xs text-gray-500">${job.price_per_hour}/hr · {job.hours_consumed?.toFixed(2)}h</p>
              </div>
              <div className="flex items-center gap-1">
                <button onClick={() => handlePause(job)} title="Pause" className="p-1.5 hover:bg-dark-border rounded text-yellow-400"><Pause size={14} /></button>
                <button onClick={() => handleUnpause(job)} title="Resume" className="p-1.5 hover:bg-dark-border rounded text-green-400"><Play size={14} /></button>
                <button onClick={() => handleStop(job)} title="Stop" className="p-1.5 hover:bg-dark-border rounded text-red-400"><Square size={14} /></button>
                <button onClick={() => handleRestart(job)} title="Restart" className="p-1.5 hover:bg-dark-border rounded text-blue-400"><RefreshCw size={14} /></button>
                <button onClick={() => handleViewLogs(job)} title="View Logs" className="p-1.5 hover:bg-dark-border rounded text-gray-400"><Terminal size={14} /></button>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* ── Running Processes ──────────────────────────────────── */}
      <div className="bg-dark-card border border-dark-border rounded-xl p-5">
        <h3 className="font-semibold mb-4">GPU Processes</h3>
        {processes.length === 0 && <p className="text-gray-500 text-sm">No processes using the GPU</p>}
        <table className="w-full text-sm">
          <thead><tr className="text-left text-gray-400 border-b border-dark-border"><th className="py-2">PID</th><th>Name</th><th>GPU</th><th>Memory</th></tr></thead>
          <tbody>
            {processes.map((p, i) => (
              <tr key={i} className="border-b border-dark-border">
                <td className="py-1 font-mono">{p.pid}</td>
                <td className="truncate max-w-[200px]">{p.name}</td>
                <td>{p.gpu_index}</td>
                <td>{p.used_gpu_memory_mb} MB</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* ── Terminal / Logs modal ──────────────────────────────── */}
      {showTerminal && selectedJob && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60" onClick={() => setShowTerminal(false)}>
          <div className="bg-dark-card border border-dark-border rounded-xl w-full max-w-3xl max-h-[80vh] overflow-hidden" onClick={e => e.stopPropagation()}>
            <div className="flex items-center justify-between p-4 border-b border-dark-border">
              <h3 className="font-semibold">Container Logs — {selectedJob.id.slice(0,8)}</h3>
              <button onClick={() => setShowTerminal(false)} className="text-gray-400 hover:text-white">✕</button>
            </div>
            <div className="p-4 bg-black rounded-b-xl overflow-y-auto max-h-[60vh] font-mono text-xs text-green-400">
              <pre className="whitespace-pre-wrap">{containerLogs || 'No logs yet...'}</pre>
              <div ref={logsEndRef} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}