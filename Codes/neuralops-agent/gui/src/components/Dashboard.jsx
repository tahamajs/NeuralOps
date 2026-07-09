import React, { useEffect, useState } from 'react';
import { Cpu, HardDrive, TrendingUp, Clock, Play, Pause } from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { api } from '../services/api';
import { wsClient } from '../services/websocket';
import { usePolling } from '../hooks/useApi';

export default function Dashboard() {
  const [agentRunning, setAgentRunning] = useState(false);
  const [balance, setBalance] = useState(null);
  const [stats, setStats] = useState(null);
  const [chartData, setChartData] = useState([]);

  // Fetch balance and stats
  const fetchData = async () => {
    try {
      const [balanceResp, statsResp, scoresResp] = await Promise.all([
        api.getBalance(),
        api.getMarketStats(),
        api.getMyScores(),
      ]);
      setBalance(balanceResp);
      setStats({
        ...statsResp,
        ...scoresResp,
        earningsToday: balanceResp?.today_earnings || 0,
      });
    } catch (e) { /* use mock data */ }
  };

  useEffect(() => { fetchData(); const t = setInterval(fetchData, 15000); return () => clearInterval(t); }, []);

  // Listen to real-time job updates
  useEffect(() => {
    const handleJobUpdate = (data) => {
      if (data.status === 'completed') {
        fetchData();
        api.getBalance().then(b => setBalance(b));
      }
    };
    wsClient.on('job_update', handleJobUpdate);
    return () => wsClient.off('job_update', handleJobUpdate);
  }, []);

  const toggleAgent = async () => {
    if (agentRunning) {
      await api.toggleListing(listingId); // need listing ID from state
    } else {
      // Start agent via IPC
    }
    setAgentRunning(!agentRunning);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">GPU Provider Dashboard</h2>
        <button
          onClick={toggleAgent}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg font-semibold ${agentRunning ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'}`}
        >
          {agentRunning ? <Pause size={18} /> : <Play size={18} />}
          {agentRunning ? 'Stop Agent' : 'Start Agent'}
        </button>
      </div>

      {/* Stats cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {[
          { icon: Cpu, label: 'GPU Usage', value: `${stats?.active_jobs || 0} jobs`, color: 'text-blue-400' },
          { icon: TrendingUp, label: 'Earnings Today', value: `$${(stats?.earningsToday || 0).toFixed(2)}`, color: 'text-green-400' },
          { icon: HardDrive, label: 'Available Balance', value: `$${(balance?.balance_usd || 0).toFixed(2)}`, color: 'text-purple-400' },
          { icon: Clock, label: 'Avg Price/hr', value: `$${stats?.avg_price_per_hour || '0.00'}`, color: 'text-orange-400' },
        ].map(({ icon: Icon, label, value, color }) => (
          <div key={label} className="bg-dark-card border border-dark-border rounded-xl p-4">
            <div className="flex items-center gap-2 mb-2"><Icon size={20} className={color} /><span className="text-gray-400 text-sm">{label}</span></div>
            <p className="text-2xl font-bold">{value}</p>
          </div>
        ))}
      </div>

      {/* Market Price Chart */}
      <div className="bg-dark-card border border-dark-border rounded-xl p-4">
        <h3 className="text-lg font-semibold mb-4">Market Price Trend (24h)</h3>
        <ResponsiveContainer width="100%" height={300}>
          <AreaChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="time" stroke="#94a3b8" />
            <YAxis stroke="#94a3b8" />
            <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }} />
            <Area type="monotone" dataKey="price" stroke="#5c7cfa" fill="#5c7cfa" fillOpacity={0.2} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}