import React, { useEffect, useState } from 'react';
import { Cpu, HardDrive, TrendingUp, Clock, Play, Pause } from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export default function Dashboard() {
  const [agentRunning, setAgentRunning] = useState(false);
  const [stats, setStats] = useState({ gpuTemp: 58, gpuUsage: 64, ramUsage: 42, earningsToday: 12.50, uptime: '3h 24m' });
  const [chartData, setChartData] = useState([
    { time: '00:00', usage: 30 }, { time: '01:00', usage: 45 }, { time: '02:00', usage: 60 }, { time: '03:00', usage: 55 },
  ]);

  const toggleAgent = async () => {
    if (agentRunning) {
      await window.neuralopsAPI.stopAgent();
    } else {
      await window.neuralopsAPI.startAgent();
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
          { icon: Cpu, label: 'GPU Usage', value: `${stats.gpuUsage}%`, color: 'text-blue-400' },
          { icon: HardDrive, label: 'RAM Usage', value: `${stats.ramUsage}%`, color: 'text-purple-400' },
          { icon: TrendingUp, label: 'Earnings Today', value: `$${stats.earningsToday}`, color: 'text-green-400' },
          { icon: Clock, label: 'Uptime', value: stats.uptime, color: 'text-orange-400' },
        ].map(({ icon: Icon, label, value, color }) => (
          <div key={label} className="bg-dark-card border border-dark-border rounded-xl p-4">
            <div className="flex items-center gap-2 mb-2">
              <Icon size={20} className={color} />
              <span className="text-gray-400 text-sm">{label}</span>
            </div>
            <p className="text-2xl font-bold">{value}</p>
          </div>
        ))}
      </div>

      {/* Chart */}
      <div className="bg-dark-card border border-dark-border rounded-xl p-4">
        <h3 className="text-lg font-semibold mb-4">GPU Utilization (24h)</h3>
        <ResponsiveContainer width="100%" height={300}>
          <AreaChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="time" stroke="#94a3b8" />
            <YAxis stroke="#94a3b8" />
            <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }} />
            <Area type="monotone" dataKey="usage" stroke="#5c7cfa" fill="#5c7cfa" fillOpacity={0.2} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}