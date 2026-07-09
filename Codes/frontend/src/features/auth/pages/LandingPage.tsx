import { Link } from "react-router-dom";
import { Terminal, Brain, Cpu, Database, ChevronRight, Sparkles, Activity, ShieldAlert, Zap } from "lucide-react";
import { useState, useEffect } from "react";

export default function LandingPage() {
  const [stats, setStats] = useState({
    gpusOnline: 1148,
    pflops: 12.8,
    jobsCompleted: 34212,
  });

  // Dynamic dashboard stats simulation
  useEffect(() => {
    const interval = setInterval(() => {
      setStats((prev) => ({
        gpusOnline: prev.gpusOnline + (Math.random() > 0.7 ? 1 : Math.random() > 0.85 ? -1 : 0),
        pflops: Number((prev.pflops + (Math.random() > 0.5 ? 0.1 : -0.1)).toFixed(1)),
        jobsCompleted: prev.jobsCompleted + 1,
      }));
    }, 4000);
    return () => clearInterval(interval);
  }, []);

  const features = [
    {
      icon: Brain,
      title: "Core Experiment Tracker",
      description: "Loss tracking, parameter visualization, log streaming, and config history in real-time.",
    },
    {
      icon: Cpu,
      title: "Optuna Sweeps & HPO",
      description: "Distributed hyperparameter sweeps with TPE, CMA-ES, and automated trial scheduling.",
    },
    {
      icon: Terminal,
      title: "Integrated Workspace",
      description: "Monaco-based collaborative LaTeX editor and floating xterm.js terminals on a single canvas.",
    },
    {
      icon: Database,
      title: "Model Registry & DVC",
      description: "Staging, production, and archiving workflow lifecycle with automatic data lineage graphs.",
    },
  ];

  return (
    <div className="min-h-screen bg-[#07080d] text-white relative overflow-hidden flex flex-col font-sans select-none">
      {/* Background Cyber Grid & Glows */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#1e293b_1px,transparent_1px),linear-gradient(to_bottom,#1e293b_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_80%_60%_at_50%_0%,#000_80%,transparent_100%)] opacity-30" />
      <div className="absolute top-[-10%] left-[10%] w-[600px] h-[600px] bg-neon-blue/15 rounded-full blur-[140px] pointer-events-none animate-pulse" />
      <div className="absolute bottom-[20%] right-[10%] w-[500px] h-[500px] bg-purple-500/10 rounded-full blur-[150px] pointer-events-none" />

      {/* Header */}
      <header className="relative z-10 max-w-7xl mx-auto w-full px-6 py-6 flex justify-between items-center border-b border-white/5 backdrop-blur-md">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-gradient-to-br from-neon-blue via-violet-600 to-purple-600 rounded-xl flex items-center justify-center shadow-[0_0_20px_rgba(59,130,246,0.5)]">
            <span className="text-white font-black text-sm">NO</span>
          </div>
          <span className="font-extrabold text-xl tracking-wider bg-clip-text text-transparent bg-gradient-to-r from-white via-slate-200 to-slate-400">
            NEURALOPS
          </span>
        </div>
        <div className="flex items-center gap-6">
          <Link to="/login" className="text-sm font-semibold text-slate-300 hover:text-neon-blue transition-colors">
            Sign In
          </Link>
          <Link
            to="/register"
            className="px-5 py-2.5 bg-gradient-to-r from-neon-blue via-violet-600 to-purple-600 hover:brightness-110 rounded-xl text-xs font-black uppercase tracking-wider transition-all duration-300 shadow-[0_0_15px_rgba(59,130,246,0.3)]"
          >
            Get Started
          </Link>
        </div>
      </header>

      {/* Hero Section */}
      <main className="relative z-10 flex-1 max-w-7xl mx-auto w-full px-6 flex flex-col justify-center py-16 text-center items-center">
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-neon-blue/10 border border-neon-blue/20 text-neon-blue text-xs font-bold uppercase tracking-widest mb-8">
          <Sparkles size={14} className="animate-spin" /> Deep Learning Cluster Orchestrator
        </div>
        <h1 className="text-5xl md:text-8xl font-black tracking-tight mb-6 max-w-5xl leading-tight bg-clip-text text-transparent bg-gradient-to-b from-white via-slate-100 to-slate-400">
          The Attention-Compression <br />
          <span className="bg-clip-text text-transparent bg-gradient-to-r from-neon-blue via-violet-400 to-purple-500 drop-shadow-[0_0_30px_rgba(59,130,246,0.2)]">
            Research Operating System
          </span>
        </h1>
        <p className="text-lg md:text-xl text-slate-400 max-w-3xl mb-12 leading-relaxed font-light">
          Manage model versioning, track hyperparameter sweeps, and schedule compute workloads globally across an intuitive, drag-and-drop glassmorphism canvas.
        </p>

        {/* Live Telemetry Banner */}
        <div className="grid grid-cols-3 gap-6 max-w-2xl w-full bg-slate-900/50 backdrop-blur-md border border-white/5 rounded-2xl p-4 mb-16 shadow-[0_4px_30px_rgba(0,0,0,0.4)]">
          <div className="flex flex-col items-center">
            <span className="text-[10px] text-slate-500 font-bold uppercase tracking-wider flex items-center gap-1.5">
              <Zap size={10} className="text-neon-blue animate-pulse" /> Live Nodes
            </span>
            <span className="text-lg md:text-2xl font-black text-neon-blue font-mono mt-1">
              {stats.gpusOnline}
            </span>
          </div>
          <div className="flex flex-col items-center border-x border-white/5">
            <span className="text-[10px] text-slate-500 font-bold uppercase tracking-wider flex items-center gap-1.5">
              <Activity size={10} className="text-purple-400" /> Throughput
            </span>
            <span className="text-lg md:text-2xl font-black text-purple-400 font-mono mt-1">
              {stats.pflops} PFLOPS
            </span>
          </div>
          <div className="flex flex-col items-center">
            <span className="text-[10px] text-slate-500 font-bold uppercase tracking-wider flex items-center gap-1.5">
              <ShieldAlert size={10} className="text-green-400" /> Jobs Run
            </span>
            <span className="text-lg md:text-2xl font-black text-green-400 font-mono mt-1">
              {stats.jobsCompleted.toLocaleString()}
            </span>
          </div>
        </div>

        {/* Hero Actions */}
        <div className="flex flex-col sm:flex-row gap-5 mb-24">
          <Link
            to="/register"
            className="px-8 py-4.5 bg-gradient-to-r from-neon-blue via-violet-600 to-purple-600 hover:brightness-110 rounded-2xl text-sm font-black uppercase tracking-widest transition-all duration-300 shadow-[0_0_25px_rgba(59,130,246,0.4)] flex items-center gap-3"
          >
            Launch Core Console <ChevronRight size={18} />
          </Link>
          <Link
            to="/login"
            className="px-8 py-4.5 bg-white/5 border border-white/10 hover:bg-white/10 rounded-2xl text-sm font-black uppercase tracking-widest transition-all duration-300 backdrop-blur-sm"
          >
            Login to Account
          </Link>
        </div>

        {/* Feature Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 w-full text-left">
          {features.map((feature, idx) => (
            <div
              key={idx}
              className="p-6 rounded-2xl border border-white/5 bg-[#0e1017]/60 backdrop-blur-md hover:border-neon-blue/30 hover:shadow-[0_0_20px_rgba(59,130,246,0.15)] transition-all duration-300 group"
            >
              <div className="w-12 h-12 rounded-xl bg-slate-900 flex items-center justify-center mb-5 group-hover:bg-neon-blue/10 group-hover:text-neon-blue transition-all duration-300 border border-white/5">
                <feature.icon size={22} className="text-slate-400 group-hover:text-neon-blue" />
              </div>
              <h3 className="font-extrabold text-white mb-2.5 text-base tracking-wide">{feature.title}</h3>
              <p className="text-sm text-slate-400 leading-relaxed font-light">{feature.description}</p>
            </div>
          ))}
        </div>
      </main>

      {/* Footer */}
      <footer className="relative z-10 max-w-7xl mx-auto w-full px-6 py-8 border-t border-white/5 flex flex-col md:flex-row justify-between items-center text-xs text-slate-500 gap-4">
        <div>© 2026 NeuralOps. All rights reserved. MIT License.</div>
        <div className="flex gap-6">
          <a href="#" className="hover:text-white transition-colors">Privacy Policy</a>
          <a href="#" className="hover:text-white transition-colors">Terms of Service</a>
          <Link to="/docs" className="hover:text-white transition-colors">Documentation</Link>
        </div>
      </footer>
    </div>
  );
}
