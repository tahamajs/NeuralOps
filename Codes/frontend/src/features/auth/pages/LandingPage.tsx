import { Link } from "react-router-dom";
import {
  Terminal,
  Brain,
  Cpu,
  Database,
  ChevronRight,
  Sparkles,
  Activity,
  ShieldAlert,
  Zap,
  Network,
  Bot,
  Layers,
  PlaySquare,
} from "lucide-react";
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
        gpusOnline:
          prev.gpusOnline +
          (Math.random() > 0.7 ? 1 : Math.random() > 0.85 ? -1 : 0),
        pflops: Number(
          (prev.pflops + (Math.random() > 0.5 ? 0.1 : -0.1)).toFixed(1),
        ),
        jobsCompleted: prev.jobsCompleted + 1,
      }));
    }, 4000);
    return () => clearInterval(interval);
  }, []);

  const features = [
    {
      icon: Brain,
      title: "Autonomous AI Agents",
      description:
        "Deploy deep research agents that iteratively propose hypotheses, run experiments, and analyze results on their own.",
    },
    {
      icon: Layers,
      title: "Floating UI Workspace",
      description:
        "A gorgeous, Windows-like floating window manager that lets you drag, drop, and resize terminal, chat, and lab windows.",
    },
    {
      icon: Network,
      title: "Marketplace AI",
      description:
        "Rent idle H100s globally or deploy your models to a decentralized network with secure ESCROW smart contracts.",
    },
    {
      icon: Database,
      title: "Model Registry & DVC",
      description:
        "Staging, production, and archiving workflow lifecycle with automatic data lineage graphs and artifact tracking.",
    },
  ];

  return (
    <div className="min-h-screen bg-[#07080d] text-white relative overflow-x-hidden flex flex-col font-sans select-none">
      {/* Background Cyber Grid & Glows */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#1e293b_1px,transparent_1px),linear-gradient(to_bottom,#1e293b_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_80%_60%_at_50%_0%,#000_80%,transparent_100%)] opacity-30" />
      <div className="absolute top-[-10%] left-[10%] w-[600px] h-[600px] bg-neon-blue/15 rounded-full blur-[140px] pointer-events-none animate-pulse" />
      <div className="absolute bottom-[20%] right-[10%] w-[500px] h-[500px] bg-purple-500/15 rounded-full blur-[150px] pointer-events-none" />
      <div className="absolute top-[40%] left-[50%] -translate-x-1/2 w-[800px] h-[300px] bg-cyan-400/10 rounded-[100%] blur-[120px] pointer-events-none" />

      {/* Header */}
      <header className="relative z-10 max-w-7xl mx-auto w-full px-6 py-6 flex justify-between items-center border-b border-white/5 backdrop-blur-md">
        <div className="flex items-center gap-3 group cursor-pointer">
          <div className="w-10 h-10 bg-gradient-to-br from-neon-blue via-cyan-400 to-purple-600 rounded-xl flex items-center justify-center shadow-[0_0_20px_rgba(59,130,246,0.5)] group-hover:shadow-[0_0_30px_rgba(59,130,246,0.8)] transition-all duration-300">
            <img
              src="/logo.jpg"
              alt="NeuralOps"
              className="w-full h-full object-cover rounded-xl"
            />
          </div>
          <span className="font-extrabold text-xl tracking-wider bg-clip-text text-transparent bg-gradient-to-r from-white via-slate-200 to-slate-400">
            NEURALOPS
          </span>
        </div>
        <div className="flex items-center gap-6">
          <Link
            to="/login"
            className="text-sm font-bold text-slate-300 hover:text-white hover:drop-shadow-[0_0_8px_rgba(255,255,255,0.8)] transition-all"
          >
            Sign In
          </Link>
          <Link
            to="/register"
            className="px-6 py-2.5 bg-white/10 hover:bg-white/20 border border-white/20 backdrop-blur-md rounded-xl text-xs font-black uppercase tracking-wider transition-all duration-300 shadow-lg hover:shadow-[0_0_20px_rgba(255,255,255,0.2)]"
          >
            Get Started
          </Link>
        </div>
      </header>

      {/* Hero Section */}
      <main className="relative z-10 flex-1 max-w-7xl mx-auto w-full px-6 flex flex-col justify-center py-20 text-center items-center">
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-cyan-500/10 border border-cyan-400/30 text-cyan-400 text-xs font-bold uppercase tracking-widest mb-8 hover:bg-cyan-500/20 transition-colors cursor-default shadow-[0_0_15px_rgba(34,211,238,0.2)]">
          <Sparkles size={14} className="animate-spin-slow" /> NeuralOps OS 2.0
          Now Live
        </div>

        <h1 className="text-6xl md:text-[5.5rem] font-black tracking-tighter mb-6 max-w-5xl leading-[1.1] bg-clip-text text-transparent bg-gradient-to-b from-white via-slate-100 to-slate-400 drop-shadow-2xl">
          The Autonomous Agent <br />
          <span className="bg-clip-text text-transparent bg-gradient-to-r from-neon-blue via-cyan-300 to-purple-500 drop-shadow-[0_0_40px_rgba(59,130,246,0.3)]">
            Operating System
          </span>
        </h1>

        <p className="text-lg md:text-xl text-slate-400 max-w-3xl mb-12 leading-relaxed font-light">
          Experience a beautiful, floating window workspace powered by
          intelligent AI Agents. Orchestrate models, run experiments, and rent
          GPU compute globally.
        </p>

        {/* Hero Actions */}
        <div className="flex flex-col sm:flex-row gap-5 mb-16">
          <Link
            to="/register"
            className="group px-8 py-4 bg-gradient-to-r from-neon-blue via-cyan-500 to-purple-600 hover:brightness-110 rounded-2xl text-sm font-black uppercase tracking-widest transition-all duration-300 shadow-[0_0_30px_rgba(59,130,246,0.5)] hover:shadow-[0_0_50px_rgba(59,130,246,0.7)] hover:-translate-y-1 flex items-center gap-3"
          >
            Launch Workspace{" "}
            <ChevronRight
              size={18}
              className="group-hover:translate-x-1 transition-transform"
            />
          </Link>
          <a
            href="#features"
            className="px-8 py-4 bg-white/5 border border-white/10 hover:bg-white/10 hover:border-white/20 rounded-2xl text-sm font-black uppercase tracking-widest transition-all duration-300 backdrop-blur-md flex items-center gap-3"
          >
            <PlaySquare size={18} /> View Features
          </a>
        </div>

        {/* Floating UI Preview Graphic */}
        <div className="relative w-full max-w-5xl h-[300px] md:h-[400px] mb-24 rounded-2xl border border-white/10 bg-slate-900/40 backdrop-blur-xl shadow-[0_20px_60px_rgba(0,0,0,0.6)] overflow-hidden group">
          {/* Mac-like Window Controls */}
          <div className="absolute top-0 w-full h-10 bg-white/5 border-b border-white/5 flex items-center px-4 gap-2">
            <div className="w-3 h-3 rounded-full bg-red-500/80" />
            <div className="w-3 h-3 rounded-full bg-yellow-500/80" />
            <div className="w-3 h-3 rounded-full bg-green-500/80" />
            <div className="mx-auto text-xs font-mono text-slate-500 font-semibold tracking-widest">
              NEURALOPS.WORKSPACE
            </div>
          </div>

          {/* Simulated Floating Windows */}
          <div className="absolute top-16 left-8 w-[350px] h-[250px] bg-slate-900/80 border border-white/10 rounded-xl backdrop-blur-2xl shadow-2xl transition-transform duration-500 group-hover:-translate-y-4 group-hover:-rotate-2">
            <div className="p-3 border-b border-white/5 flex items-center gap-2 text-xs font-bold text-neon-blue">
              <Bot size={14} /> Agent Chat
            </div>
            <div className="p-4 space-y-3">
              <div className="w-3/4 h-8 rounded-lg bg-white/5" />
              <div className="w-full h-12 rounded-lg bg-neon-blue/10 border border-neon-blue/20" />
              <div className="w-1/2 h-8 rounded-lg bg-white/5" />
            </div>
          </div>

          <div className="absolute top-24 right-8 w-[400px] h-[220px] bg-slate-900/80 border border-white/10 rounded-xl backdrop-blur-2xl shadow-2xl transition-transform duration-500 group-hover:translate-y-4 group-hover:rotate-2">
            <div className="p-3 border-b border-white/5 flex items-center gap-2 text-xs font-bold text-purple-400">
              <Terminal size={14} /> xterm.js Cluster
            </div>
            <div className="p-4 space-y-2">
              <div className="w-full h-4 rounded bg-purple-500/20" />
              <div className="w-5/6 h-4 rounded bg-white/5" />
              <div className="w-4/6 h-4 rounded bg-white/5" />
              <div className="w-1/2 h-4 rounded bg-green-500/20 mt-4" />
            </div>
          </div>
        </div>

        {/* Live Telemetry Banner */}
        <div className="grid grid-cols-3 gap-6 max-w-3xl w-full bg-slate-900/40 backdrop-blur-xl border border-white/10 rounded-3xl p-6 mb-20 shadow-[0_8px_40px_rgba(0,0,0,0.5)]">
          <div className="flex flex-col items-center">
            <span className="text-[10px] text-slate-400 font-bold uppercase tracking-widest flex items-center gap-2 mb-2">
              <Zap size={12} className="text-neon-blue animate-pulse" /> Network
              Nodes
            </span>
            <span className="text-2xl md:text-4xl font-black text-white font-mono drop-shadow-[0_0_10px_rgba(255,255,255,0.3)]">
              {stats.gpusOnline}
            </span>
          </div>
          <div className="flex flex-col items-center border-x border-white/10">
            <span className="text-[10px] text-slate-400 font-bold uppercase tracking-widest flex items-center gap-2 mb-2">
              <Activity size={12} className="text-purple-400" /> Total
              Throughput
            </span>
            <span className="text-2xl md:text-4xl font-black text-white font-mono drop-shadow-[0_0_10px_rgba(255,255,255,0.3)]">
              {stats.pflops}{" "}
              <span className="text-lg text-slate-500">PFLOPS</span>
            </span>
          </div>
          <div className="flex flex-col items-center">
            <span className="text-[10px] text-slate-400 font-bold uppercase tracking-widest flex items-center gap-2 mb-2">
              <ShieldAlert size={12} className="text-cyan-400" /> Jobs Executed
            </span>
            <span className="text-2xl md:text-4xl font-black text-white font-mono drop-shadow-[0_0_10px_rgba(255,255,255,0.3)]">
              {stats.jobsCompleted.toLocaleString()}
            </span>
          </div>
        </div>

        {/* Feature Grid */}
        <div
          id="features"
          className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 w-full text-left"
        >
          {features.map((feature, idx) => (
            <div
              key={idx}
              className="p-8 rounded-3xl border border-white/10 bg-slate-900/40 backdrop-blur-xl hover:bg-slate-900/60 hover:border-cyan-400/40 hover:shadow-[0_0_30px_rgba(34,211,238,0.15)] hover:-translate-y-2 transition-all duration-300 group"
            >
              <div className="w-14 h-14 rounded-2xl bg-white/5 flex items-center justify-center mb-6 group-hover:bg-gradient-to-br group-hover:from-neon-blue group-hover:to-purple-600 transition-all duration-500 border border-white/10">
                <feature.icon
                  size={26}
                  className="text-slate-300 group-hover:text-white transition-colors"
                />
              </div>
              <h3 className="font-extrabold text-white mb-3 text-lg tracking-wide">
                {feature.title}
              </h3>
              <p className="text-sm text-slate-400 leading-relaxed font-light">
                {feature.description}
              </p>
            </div>
          ))}
        </div>
      </main>

      {/* Footer */}
      <footer className="relative z-10 max-w-7xl mx-auto w-full px-6 py-10 border-t border-white/10 flex flex-col md:flex-row justify-between items-center text-xs text-slate-500 font-medium">
        <div>© 2026 NeuralOps Inc. Powered by Glassmorphism & Agents.</div>
        <div className="flex gap-8 mt-4 md:mt-0">
          <a
            href="#"
            className="hover:text-white transition-colors uppercase tracking-widest"
          >
            Privacy
          </a>
          <a
            href="#"
            className="hover:text-white transition-colors uppercase tracking-widest"
          >
            Terms
          </a>
          <a
            href="#"
            className="hover:text-white transition-colors uppercase tracking-widest"
          >
            Docs
          </a>
        </div>
      </footer>
    </div>
  );
}
