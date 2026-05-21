import React, { useState } from 'react';
import { DollarSign, TrendingUp, TrendingDown, Wallet, Download, ArrowUpRight } from 'lucide-react';
import { BarChart, Bar, LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

const earningHistory = [
  { date: 'Mon', amount: 3.20 }, { date: 'Tue', amount: 5.40 }, { date: 'Wed', amount: 2.80 },
  { date: 'Thu', amount: 7.10 }, { date: 'Fri', amount: 4.50 }, { date: 'Sat', amount: 6.30 }, { date: 'Sun', amount: 3.90 },
];

const monthlyData = [
  { month: 'Jan', earnings: 98 }, { month: 'Feb', earnings: 142 }, { month: 'Mar', earnings: 187 },
  { month: 'Apr', earnings: 210 }, { month: 'May', earnings: 168 }, { month: 'Jun', earnings: 245 },
];

const jobEarnings = [
  { name: 'ResNet Training', value: 45 }, { name: 'BERT Fine-tune', value: 28 },
  { name: 'Stable Diffusion', value: 18 }, { name: 'Other', value: 9 },
];
const COLORS = ['#5c7cfa', '#22d3ee', '#f59e0b', '#a78bfa'];

export default function Earnings() {
  const [period, setPeriod] = useState('week');

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Earnings</h2>
        <div className="flex gap-2 bg-dark-card border border-dark-border rounded-lg p-1">
          {['week', 'month', 'year'].map(p => (
            <button key={p} onClick={() => setPeriod(p)}
              className={`px-3 py-1 rounded-md text-sm font-medium transition-colors ${period === p ? 'bg-primary-600 text-white' : 'text-gray-400 hover:text-white'}`}>
              {p.charAt(0).toUpperCase() + p.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Balance Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {[
          { icon: Wallet, label: 'Available Balance', value: '$156.42', color: 'text-green-400', bg: 'bg-green-400/10' },
          { icon: TrendingUp, label: 'Earnings This Week', value: '$33.20', sub: '+12.5%', color: 'text-blue-400', bg: 'bg-blue-400/10' },
          { icon: DollarSign, label: 'Lifetime Earnings', value: '$1,247.80', color: 'text-purple-400', bg: 'bg-purple-400/10' },
        ].map(({ icon: Icon, label, value, sub, color, bg }) => (
          <div key={label} className="bg-dark-card border border-dark-border rounded-xl p-5">
            <div className="flex items-center justify-between mb-3">
              <span className="text-gray-400 text-sm">{label}</span>
              <div className={`p-2 rounded-lg ${bg}`}><Icon size={18} className={color} /></div>
            </div>
            <p className="text-3xl font-bold">{value}</p>
            {sub && <p className="text-green-400 text-sm mt-1">{sub} vs last week</p>}
          </div>
        ))}
      </div>

      {/* Main Chart */}
      <div className="bg-dark-card border border-dark-border rounded-xl p-5">
        <h3 className="text-lg font-semibold mb-4">Earnings Overview</h3>
        <ResponsiveContainer width="100%" height={300}>
          {period === 'week' ? (
            <AreaChart data={earningHistory}>
              <defs><linearGradient id="earnGrad" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#5c7cfa" stopOpacity={0.3} /><stop offset="95%" stopColor="#5c7cfa" stopOpacity={0} /></linearGradient></defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="date" stroke="#94a3b8" /><YAxis stroke="#94a3b8" />
              <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }} />
              <Area type="monotone" dataKey="amount" stroke="#5c7cfa" fill="url(#earnGrad)" />
            </AreaChart>
          ) : (
            <BarChart data={monthlyData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="month" stroke="#94a3b8" /><YAxis stroke="#94a3b8" />
              <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }} />
              <Bar dataKey="earnings" fill="#5c7cfa" radius={[4, 4, 0, 0]} />
            </BarChart>
          )}
        </ResponsiveContainer>
      </div>

      {/* Bottom Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-dark-card border border-dark-border rounded-xl p-5">
          <h3 className="text-lg font-semibold mb-4">Earnings by Job Type</h3>
          <ResponsiveContainer width="100%" height={200}>
            <PieChart><Pie data={jobEarnings} cx="50%" cy="50%" innerRadius={50} outerRadius={80} paddingAngle={5} dataKey="value">
              {jobEarnings.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie><Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }} /></PieChart>
          </ResponsiveContainer>
          <div className="flex justify-center gap-4 mt-2">{jobEarnings.map((e, i) => (
            <div key={e.name} className="flex items-center gap-1 text-sm"><div className="w-3 h-3 rounded-full" style={{ background: COLORS[i] }} />{e.name}</div>
          ))}</div>
        </div>
        <div className="bg-dark-card border border-dark-border rounded-xl p-5">
          <h3 className="text-lg font-semibold mb-4">Quick Actions</h3>
          <div className="space-y-3">
            <button className="w-full flex items-center justify-between bg-primary-600 hover:bg-primary-700 px-4 py-3 rounded-lg font-medium transition-colors">
              <span className="flex items-center gap-2"><Download size={18} /> Request Payout</span>
              <ArrowUpRight size={16} />
            </button>
            <button className="w-full flex items-center justify-between bg-dark-bg hover:bg-dark-border border border-dark-border px-4 py-3 rounded-lg font-medium transition-colors text-gray-300">
              <span>View Transaction History</span>
              <ArrowUpRight size={16} />
            </button>
            <div className="bg-dark-bg border border-dark-border rounded-lg p-4 mt-3">
              <p className="text-gray-400 text-sm">Next payout threshold</p>
              <div className="flex items-center justify-between mt-1">
                <span className="text-lg font-semibold">$156.42 / $200.00</span>
                <span className="text-sm text-gray-500">78%</span>
              </div>
              <div className="w-full bg-dark-border rounded-full h-2 mt-2">
                <div className="bg-primary-600 h-2 rounded-full" style={{ width: '78%' }} />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}