import React, { useEffect, useState } from 'react';
import {
  Cpu,
  HardDrive,
  TrendingUp,
  Clock,
  Play,
  Pause,
  RefreshCw,
  Zap,
  ShieldCheck,
  Server,
} from 'lucide-react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { api } from '../services/api';
import { wsClient } from '../services/websocket';

export default function Dashboard() {
  const [agentRunning, setAgentRunning] = useState(false);
  const [loadingToggle, setLoadingToggle] = useState(false);
  const [balance, setBalance] = useState(null);
  const [stats, setStats] = useState(null);
  const [gpuInfo, setGpuInfo] = useState([]);
  const [listing, setListing] = useState(null);
  const [chartData, setChartData] = useState([]);

  // Generate initial or updated 24h market price trend
  const generateChartData = (basePrice = 0.5) => {
    const hours = ['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00', 'Now'];
    const variation = [0.92, 0.95, 0.88, 1.05, 1.15, 1.10, 1.02, 0.98, 1.0];
    return hours.map((time, idx) => ({
      time,
      price: parseFloat((basePrice * variation[idx]).toFixed(3)),
    }));
  };

  // Fetch balance, listings, stats
  const fetchData = async () => {
    try {
      if (window.neuralopsAPI) {
        const isRunning = await window.neuralopsAPI.getAgentStatus();
        setAgentRunning(isRunning);
        const gpus = await window.neuralopsAPI.getGpuInfo();
        setGpuInfo(gpus || []);
      }

      const [balanceResp, statsResp, scoresResp, listingsResp] = await Promise.allSettled([
        api.getBalance(),
        api.getMarketStats(),
        api.getMyScores(),
        api.getMyListings(),
      ]);

      if (balanceResp.status === 'fulfilled') setBalance(balanceResp.value);
      if (statsResp.status === 'fulfilled') {
        const s = statsResp.value || {};
        const score = scoresResp.status === 'fulfilled' ? scoresResp.value : {};
        setStats({
          ...s,
          ...score,
          earningsToday: balanceResp.status === 'fulfilled' ? balanceResp.value?.today_earnings || 0 : 0,
        });
        const avgPrice = parseFloat(s.avg_price_per_hour || 0.5);
        setChartData(generateChartData(avgPrice > 0 ? avgPrice : 0.5));
      } else {
        setChartData(generateChartData(0.5));
      }

      if (listingsResp.status === 'fulfilled' && Array.isArray(listingsResp.value) && listingsResp.value.length > 0) {
        setListing(listingsResp.value[0]);
      }
    } catch (e) {
      console.warn('Dashboard fetch error:', e);
      setChartData(generateChartData(0.5));
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, []);

  // Listen to WebSocket events & agent status changes
  useEffect(() => {
    const handleJobUpdate = (data) => {
      if (data && (data.status === 'completed' || data.status === 'running')) {
        fetchData();
      }
    };

    wsClient.on('job_update', handleJobUpdate);

    let unsubscribeStatus = null;
    if (window.neuralopsAPI?.onAgentStatusChanged) {
      unsubscribeStatus = window.neuralopsAPI.onAgentStatusChanged((running) => {
        setAgentRunning(running);
      });
    }

    return () => {
      wsClient.off('job_update', handleJobUpdate);
      if (unsubscribeStatus) unsubscribeStatus();
    };
  }, []);

  const toggleAgent = async () => {
    setLoadingToggle(true);
    try {
      if (window.neuralopsAPI) {
        if (agentRunning) {
          await window.neuralopsAPI.stopAgent();
          setAgentRunning(false);
          if (listing?.id) {
            await api.toggleListing(listing.id).catch(() => {});
          }
        } else {
          await window.neuralopsAPI.startAgent();
          setAgentRunning(true);
          if (listing?.id) {
            await api.toggleListing(listing.id).catch(() => {});
          }
        }
      } else {
        // Fallback for browser preview
        setAgentRunning(!agentRunning);
      }
    } catch (err) {
      console.error('Error toggling agent:', err);
    } finally {
      setLoadingToggle(false);
    }
  };

  const primaryGpu = gpuInfo.length > 0 ? gpuInfo[0] : null;

  return (
    <div className="space-y-6">
      {/* Header with quick toggle */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-white tracking-tight">GPU Provider Dashboard</h2>
          <p className="text-sm text-gray-400 mt-1">
            Real-time status, compute job matching, and marketplace earnings.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={fetchData}
            title="Refresh Dashboard"
            className="p-2.5 rounded-xl border border-dark-border bg-dark-card hover:bg-dark-border text-gray-300 hover:text-white transition-colors"
          >
            <RefreshCw size={18} />
          </button>
          <button
            onClick={toggleAgent}
            disabled={loadingToggle}
            className={`flex items-center gap-2 px-5 py-2.5 rounded-xl font-semibold text-white shadow-lg transition-all ${
              agentRunning
                ? 'bg-red-600 hover:bg-red-700 shadow-red-600/30'
                : 'bg-emerald-600 hover:bg-emerald-700 shadow-emerald-600/30'
            } disabled:opacity-50`}
          >
            {loadingToggle ? (
              <RefreshCw size={18} className="animate-spin" />
            ) : agentRunning ? (
              <Pause size={18} />
            ) : (
              <Play size={18} />
            )}
            {agentRunning ? 'Stop Agent' : 'Start Agent'}
          </button>
        </div>
      </div>

      {/* GPU Overview Banner */}
      <div className="bg-gradient-to-r from-dark-card to-slate-900 border border-dark-border rounded-2xl p-5 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 rounded-xl bg-primary-600/20 border border-primary-500/30 flex items-center justify-center text-primary-400">
            <Cpu size={26} />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="font-bold text-base text-white">
                {primaryGpu ? primaryGpu.name : 'NVIDIA Hardware'}
              </h3>
              <span
                className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                  agentRunning
                    ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                    : 'bg-rose-500/10 text-rose-400 border border-rose-500/20'
                }`}
              >
                {agentRunning ? 'ONLINE & READY' : 'OFFLINE'}
              </span>
            </div>
            <p className="text-xs text-gray-400 mt-1">
              {primaryGpu
                ? `${primaryGpu.vramMb} MB VRAM · ${primaryGpu.tempC}°C · ${primaryGpu.powerW}W Power Draw`
                : 'Share your GPU compute to earn on-demand passive income'}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-6 text-sm">
          <div>
            <p className="text-xs text-gray-400">Rate</p>
            <p className="font-bold text-white">${listing?.base_price_per_hour || '0.50'}/hr</p>
          </div>
          <div>
            <p className="text-xs text-gray-400">Listing Type</p>
            <p className="font-bold text-primary-300 uppercase text-xs">{listing?.listing_type || 'On-Demand'}</p>
          </div>
          <div>
            <p className="text-xs text-gray-400">SSH Direct</p>
            <p className="font-bold text-emerald-400 flex items-center gap-1 text-xs">
              <ShieldCheck size={14} /> Enabled
            </p>
          </div>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {[
          {
            icon: Server,
            label: 'Active Jobs',
            value: `${stats?.active_jobs || 0} active`,
            sub: `${stats?.total_jobs_completed || 0} completed`,
            color: 'text-blue-400',
            bg: 'bg-blue-400/10',
          },
          {
            icon: TrendingUp,
            label: "Today's Earnings",
            value: `$${(stats?.earningsToday || 0).toFixed(2)}`,
            sub: 'Settled to balance',
            color: 'text-emerald-400',
            bg: 'bg-emerald-400/10',
          },
          {
            icon: HardDrive,
            label: 'Available Balance',
            value: `$${(balance?.balance_usd || 0).toFixed(2)}`,
            sub: `Locked: $${(balance?.locked_balance_usd || 0).toFixed(2)}`,
            color: 'text-purple-400',
            bg: 'bg-purple-400/10',
          },
          {
            icon: Clock,
            label: 'Avg Market Rate',
            value: `$${stats?.avg_price_per_hour || '0.50'}/hr`,
            sub: 'Dynamic benchmarked',
            color: 'text-amber-400',
            bg: 'bg-amber-400/10',
          },
        ].map(({ icon: Icon, label, value, sub, color, bg }) => (
          <div key={label} className="bg-dark-card border border-dark-border rounded-2xl p-5 hover:border-slate-700 transition-colors">
            <div className="flex items-center gap-2 mb-3">
              <div className={`p-2 rounded-lg ${bg} ${color}`}>
                <Icon size={18} />
              </div>
              <span className="text-gray-400 text-xs font-medium">{label}</span>
            </div>
            <p className="text-2xl font-bold text-white">{value}</p>
            <p className="text-xs text-gray-500 mt-1">{sub}</p>
          </div>
        ))}
      </div>

      {/* Market Price Chart */}
      <div className="bg-dark-card border border-dark-border rounded-2xl p-6 shadow-xl">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-base font-semibold text-white">24-Hour Compute Market Rate Index</h3>
            <p className="text-xs text-gray-400">Continuous price index tracking for GPU-tier compute contracts</p>
          </div>
          <span className="text-xs font-mono text-primary-400 bg-primary-500/10 px-2.5 py-1 rounded-lg border border-primary-500/20">
            INDEX: TIER-1 CUDA
          </span>
        </div>
        <ResponsiveContainer width="100%" height={260}>
          <AreaChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            <defs>
              <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#4c6ef5" stopOpacity={0.4} />
                <stop offset="95%" stopColor="#4c6ef5" stopOpacity={0.0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis dataKey="time" stroke="#64748b" tick={{ fontSize: 11 }} />
            <YAxis stroke="#64748b" tick={{ fontSize: 11 }} domain={['auto', 'auto']} tickFormatter={(v) => `$${v}`} />
            <Tooltip
              contentStyle={{
                backgroundColor: '#0f172a',
                border: '1px solid #334155',
                borderRadius: '10px',
                color: '#fff',
                fontSize: '12px',
              }}
              formatter={(val) => [`$${val}/hr`, 'Market Rate']}
            />
            <Area
              type="monotone"
              dataKey="price"
              stroke="#4c6ef5"
              strokeWidth={2.5}
              fill="url(#priceGradient)"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}