import { Link } from "react-router-dom";
import { Terminal, Brain, Cpu, Database, ChevronRight, Sparkles } from "lucide-react";

export default function LandingPage() {
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
    <div className="min-h-screen bg-slate-950 text-white relative overflow-hidden flex flex-col font-sans">
      {/* Background Cyber Grid & Glows */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#0f172a_1px,transparent_1px),linear-gradient(to_bottom,#0f172a_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,#000_70%,transparent_100%)] opacity-60" />
      <div className="absolute top-0 left-1/4 w-[500px] h-[500px] bg-neon-blue/10 rounded-full blur-[120px] pointer-events-none" />
      <div className="absolute top-1/3 right-1/4 w-[600px] h-[600px] bg-purple-500/5 rounded-full blur-[150px] pointer-events-none" />

      {/* Header */}
      <header className="relative z-10 max-w-7xl mx-auto w-full px-6 py-6 flex justify-between items-center border-b border-slate-900/80 backdrop-blur-md">
        <div className="flex items-center gap-2">
          <div className="w-9 h-9 bg-gradient-to-br from-neon-blue to-purple-500 rounded-xl flex items-center justify-center shadow-glow-sm">
            <span className="text-white font-black text-sm">NO</span>
          </div>
          <span className="font-bold text-lg tracking-widest bg-clip-text text-transparent bg-gradient-to-r from-white to-slate-400">
            NEURALOPS
          </span>
        </div>
        <div className="flex items-center gap-4">
          <Link to="/login" className="text-sm font-semibold hover:text-neon-blue transition-colors">
            Sign In
          </Link>
          <Link
            to="/register"
            className="px-4 py-2 bg-gradient-to-r from-neon-blue to-purple-600 hover:from-neon-blue/90 hover:to-purple-600/90 rounded-xl text-sm font-black uppercase tracking-wider transition-all duration-300 shadow-glow-sm"
          >
            Get Started
          </Link>
        </div>
      </header>

      {/* Hero Section */}
      <main className="relative z-10 flex-1 max-w-7xl mx-auto w-full px-6 flex flex-col justify-center py-20 text-center items-center">
        <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-neon-blue/10 border border-neon-blue/20 text-neon-blue text-xs font-semibold uppercase tracking-widest mb-6 animate-pulse">
          <Sparkles size={12} /> Privacy-First Self-Hosted Platform
        </div>
        <h1 className="text-5xl md:text-7xl font-black tracking-tight mb-6 max-w-4xl bg-clip-text text-transparent bg-gradient-to-b from-white to-slate-300">
          The Attention-Compression <br />
          <span className="bg-clip-text text-transparent bg-gradient-to-r from-neon-blue via-violet-400 to-purple-500">
            Research OS
          </span>
        </h1>
        <p className="text-lg text-slate-400 max-w-2xl mb-10 leading-relaxed">
          Track experiments, scale hyperparameter sweeps, edit papers collaboratively, and manage model versioning—all from a stunning, responsive cyber-glass desktop interface.
        </p>

        {/* Hero Actions */}
        <div className="flex flex-col sm:flex-row gap-4 mb-20">
          <Link
            to="/register"
            className="px-8 py-4 bg-gradient-to-r from-neon-blue to-purple-600 hover:from-neon-blue/90 hover:to-purple-600/90 rounded-2xl text-base font-black uppercase tracking-widest transition-all duration-300 shadow-glow-md flex items-center gap-2"
          >
            Launch Console <ChevronRight size={18} />
          </Link>
          <Link
            to="/login"
            className="px-8 py-4 bg-slate-900 border border-slate-800 hover:bg-slate-800 rounded-2xl text-base font-black uppercase tracking-widest transition-all duration-300"
          >
            Login
          </Link>
        </div>

        {/* Feature Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 w-full text-left">
          {features.map((feature, idx) => (
            <div
              key={idx}
              className="p-6 rounded-2xl border border-slate-900 bg-slate-950/40 backdrop-blur-sm hover:border-slate-800/80 transition-all duration-300 group"
            >
              <div className="w-10 h-10 rounded-xl bg-slate-900 flex items-center justify-center mb-4 group-hover:bg-neon-blue/10 group-hover:text-neon-blue transition-all duration-300">
                <feature.icon size={20} className="text-slate-400 group-hover:text-neon-blue" />
              </div>
              <h3 className="font-bold text-white mb-2">{feature.title}</h3>
              <p className="text-sm text-slate-400 leading-relaxed">{feature.description}</p>
            </div>
          ))}
        </div>
      </main>

      {/* Footer */}
      <footer className="relative z-10 max-w-7xl mx-auto w-full px-6 py-8 border-t border-slate-900/80 flex flex-col md:flex-row justify-between items-center text-xs text-slate-500 gap-4">
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
