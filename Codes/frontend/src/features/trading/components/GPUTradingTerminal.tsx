import React, { useState, useEffect } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import {
  Activity,
  Cpu,
  Server,
  Zap,
  ChevronUp,
  ChevronDown,
  DollarSign,
} from "lucide-react";

const mockData = Array.from({ length: 24 }).map((_, i) => ({
  time: `${i}:00`,
  price: 2.45 + Math.random() * 0.5 + Math.sin(i / 3) * 0.2,
  volume: Math.floor(Math.random() * 1000) + 500,
}));

export default function GPUTradingTerminal() {
  const [activeTab, setActiveTab] = useState("chart");
  const [selectedGpu, setSelectedGpu] = useState("A100-80GB");

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 p-6 font-sans antialiased overflow-hidden selection:bg-indigo-500/30">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <header className="flex flex-col md:flex-row justify-between items-start md:items-center p-6 bg-slate-900/50 backdrop-blur-xl border border-slate-800/60 rounded-2xl shadow-2xl">
          <div>
            <h1 className="text-3xl font-extrabold bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400 bg-clip-text text-transparent tracking-tight">
              NeuralOps GPU Exchange
            </h1>
            <p className="text-slate-400 mt-1 text-sm font-medium">
              Real-time Spot & Spot-Interruptible Compute
            </p>
          </div>
          <div className="flex items-center gap-4 mt-4 md:mt-0">
            <div className="flex flex-col items-end">
              <span className="text-xs text-slate-500 uppercase tracking-wider font-semibold">
                24h Vol
              </span>
              <span className="text-lg font-bold text-slate-200">
                145.2K H/s
              </span>
            </div>
            <div className="h-10 w-px bg-slate-800 hidden md:block"></div>
            <div className="flex flex-col items-end">
              <span className="text-xs text-slate-500 uppercase tracking-wider font-semibold">
                Network Load
              </span>
              <span className="text-lg font-bold text-emerald-400 flex items-center gap-1">
                <Activity size={16} className="animate-pulse" /> 84%
              </span>
            </div>
          </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main Chart Area */}
          <div className="lg:col-span-2 space-y-6">
            {/* KPI Cards */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              {[
                {
                  label: "A100-80GB Spot",
                  price: "$2.85/hr",
                  change: "+12.4%",
                  up: true,
                  icon: <Cpu size={18} />,
                },
                {
                  label: "H100 PCIe",
                  price: "$4.12/hr",
                  change: "+5.2%",
                  up: true,
                  icon: <Zap size={18} />,
                },
                {
                  label: "RTX 4090",
                  price: "$0.45/hr",
                  change: "-2.1%",
                  up: false,
                  icon: <Server size={18} />,
                },
                {
                  label: "V100-32GB",
                  price: "$1.15/hr",
                  change: "-8.4%",
                  up: false,
                  icon: <Cpu size={18} />,
                },
              ].map((kpi, idx) => (
                <div
                  key={idx}
                  className="group p-5 bg-slate-900/40 backdrop-blur-md border border-slate-800/50 rounded-2xl hover:bg-slate-800/60 transition-all duration-300 cursor-pointer hover:-translate-y-1 hover:shadow-[0_0_20px_rgba(99,102,241,0.1)]"
                >
                  <div className="flex justify-between items-start mb-2">
                    <span className="text-slate-400 text-sm font-medium flex items-center gap-2">
                      <span className="p-1.5 bg-slate-800/80 rounded-lg text-indigo-400 group-hover:text-indigo-300 transition-colors">
                        {kpi.icon}
                      </span>
                    </span>
                    <span
                      className={`flex items-center text-xs font-bold px-2 py-1 rounded-full ${kpi.up ? "bg-emerald-500/10 text-emerald-400" : "bg-rose-500/10 text-rose-400"}`}
                    >
                      {kpi.up ? (
                        <ChevronUp size={14} className="mr-0.5" />
                      ) : (
                        <ChevronDown size={14} className="mr-0.5" />
                      )}
                      {kpi.change}
                    </span>
                  </div>
                  <div className="text-slate-300 text-xs font-medium mb-1">
                    {kpi.label}
                  </div>
                  <div className="text-2xl font-bold text-white tracking-tight">
                    {kpi.price}
                  </div>
                </div>
              ))}
            </div>

            {/* Chart Widget */}
            <div className="p-6 bg-slate-900/60 backdrop-blur-xl border border-slate-800/60 rounded-3xl shadow-xl relative overflow-hidden">
              <div className="absolute top-0 left-0 w-full h-px bg-gradient-to-r from-transparent via-indigo-500/50 to-transparent"></div>

              <div className="flex justify-between items-center mb-6">
                <div className="flex items-center gap-3">
                  <select
                    className="bg-slate-950/50 border border-slate-800 text-slate-200 text-sm rounded-xl px-4 py-2 focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 outline-none transition-all"
                    value={selectedGpu}
                    onChange={(e) => setSelectedGpu(e.target.value)}
                  >
                    <option value="A100-80GB">NVIDIA A100 (80GB)</option>
                    <option value="H100">NVIDIA H100</option>
                    <option value="RTX4090">NVIDIA RTX 4090</option>
                  </select>
                </div>
                <div className="flex bg-slate-950/50 rounded-lg p-1 border border-slate-800/80">
                  {["1H", "1D", "1W", "1M"].map((t) => (
                    <button
                      key={t}
                      className={`px-3 py-1 text-xs font-semibold rounded-md transition-all ${t === "1D" ? "bg-indigo-600 text-white shadow-md" : "text-slate-400 hover:text-slate-200 hover:bg-slate-800"}`}
                    >
                      {t}
                    </button>
                  ))}
                </div>
              </div>

              <div className="h-[350px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={mockData}
                    margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
                  >
                    <defs>
                      <linearGradient
                        id="colorPrice"
                        x1="0"
                        y1="0"
                        x2="0"
                        y2="1"
                      >
                        <stop
                          offset="5%"
                          stopColor="#818cf8"
                          stopOpacity={0.3}
                        />
                        <stop
                          offset="95%"
                          stopColor="#818cf8"
                          stopOpacity={0}
                        />
                      </linearGradient>
                    </defs>
                    <CartesianGrid
                      strokeDasharray="3 3"
                      stroke="#334155"
                      vertical={false}
                    />
                    <XAxis
                      dataKey="time"
                      stroke="#64748b"
                      tick={{ fill: "#64748b", fontSize: 12 }}
                      tickLine={false}
                      axisLine={false}
                    />
                    <YAxis
                      stroke="#64748b"
                      tick={{ fill: "#64748b", fontSize: 12 }}
                      tickLine={false}
                      axisLine={false}
                      tickFormatter={(v) => `$${v.toFixed(2)}`}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "#0f172a",
                        borderColor: "#334155",
                        borderRadius: "12px",
                        boxShadow: "0 10px 15px -3px rgba(0, 0, 0, 0.5)",
                      }}
                      itemStyle={{ color: "#e2e8f0", fontWeight: "bold" }}
                      labelStyle={{ color: "#94a3b8", marginBottom: "4px" }}
                    />
                    <Line
                      type="monotone"
                      dataKey="price"
                      stroke="#818cf8"
                      strokeWidth={3}
                      dot={false}
                      activeDot={{
                        r: 6,
                        fill: "#818cf8",
                        stroke: "#1e293b",
                        strokeWidth: 2,
                      }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Sidebar Action Panel */}
          <div className="space-y-6">
            <div className="p-6 bg-slate-900/60 backdrop-blur-xl border border-slate-800/60 rounded-3xl shadow-xl relative">
              <div className="absolute top-0 right-0 w-32 h-32 bg-indigo-500/10 rounded-full blur-3xl pointer-events-none"></div>

              <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
                <DollarSign size={20} className="text-emerald-400" /> Place
                Order
              </h3>

              <div className="flex rounded-xl bg-slate-950/50 p-1 mb-6 border border-slate-800/80">
                <button className="flex-1 py-2 text-sm font-semibold rounded-lg bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                  Buy Compute
                </button>
                <button className="flex-1 py-2 text-sm font-semibold rounded-lg text-slate-400 hover:text-slate-200 transition-colors">
                  Sell Compute
                </button>
              </div>

              <div className="space-y-4">
                <div>
                  <label className="block text-xs font-medium text-slate-400 mb-1.5 uppercase tracking-wider">
                    Instance Type
                  </label>
                  <select className="w-full bg-slate-950/80 border border-slate-800 text-slate-200 text-sm rounded-xl px-4 py-3 focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 outline-none transition-all">
                    <option>A100-80GB (Spot)</option>
                    <option>H100 (Spot)</option>
                    <option>RTX 4090 (Reserved)</option>
                  </select>
                </div>

                <div>
                  <label className="block text-xs font-medium text-slate-400 mb-1.5 uppercase tracking-wider">
                    Max Bid Price ($/hr)
                  </label>
                  <div className="relative">
                    <span className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-500">
                      $
                    </span>
                    <input
                      type="number"
                      defaultValue="2.85"
                      step="0.01"
                      className="w-full bg-slate-950/80 border border-slate-800 text-white font-medium text-lg rounded-xl pl-8 pr-4 py-3 focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 outline-none transition-all"
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-xs font-medium text-slate-400 mb-1.5 uppercase tracking-wider">
                    Quantity (GPUs)
                  </label>
                  <input
                    type="number"
                    defaultValue="8"
                    className="w-full bg-slate-950/80 border border-slate-800 text-white font-medium text-lg rounded-xl px-4 py-3 focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 outline-none transition-all"
                  />
                </div>
              </div>

              <button className="w-full mt-8 bg-gradient-to-r from-emerald-500 to-emerald-400 hover:from-emerald-400 hover:to-emerald-300 text-slate-950 font-bold py-3.5 px-4 rounded-xl shadow-[0_0_20px_rgba(16,185,129,0.3)] hover:shadow-[0_0_25px_rgba(16,185,129,0.5)] transition-all transform hover:-translate-y-0.5 active:translate-y-0">
                Submit Bid Order
              </button>
            </div>

            {/* Live Order Book */}
            <div className="p-6 bg-slate-900/60 backdrop-blur-xl border border-slate-800/60 rounded-3xl shadow-xl">
              <h3 className="text-sm font-bold text-white mb-4 uppercase tracking-wider flex items-center justify-between">
                Order Book
                <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse"></span>
              </h3>

              <div className="space-y-1 text-sm font-mono">
                <div className="flex justify-between text-slate-500 text-xs mb-2">
                  <span>Price</span>
                  <span>GPUs</span>
                </div>
                {/* Asks (Sell Orders) */}
                <div className="space-y-0.5 mb-4">
                  {[2.89, 2.88, 2.87, 2.86].map((p) => (
                    <div
                      key={p}
                      className="flex justify-between items-center group cursor-pointer hover:bg-slate-800/50 px-1 rounded"
                    >
                      <span className="text-rose-400">${p}</span>
                      <span className="text-slate-300">
                        {Math.floor(Math.random() * 32) + 4}
                      </span>
                    </div>
                  ))}
                </div>

                <div className="py-2 border-y border-slate-800 flex justify-between items-center bg-slate-900/50">
                  <span className="text-lg font-bold text-white">$2.85</span>
                  <span className="text-xs text-emerald-400 flex items-center">
                    <ChevronUp size={14} /> 0.12
                  </span>
                </div>

                {/* Bids (Buy Orders) */}
                <div className="space-y-0.5 mt-4">
                  {[2.85, 2.84, 2.83, 2.82].map((p) => (
                    <div
                      key={p}
                      className="flex justify-between items-center group cursor-pointer hover:bg-slate-800/50 px-1 rounded"
                    >
                      <span className="text-emerald-400">${p}</span>
                      <span className="text-slate-300">
                        {Math.floor(Math.random() * 64) + 8}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
