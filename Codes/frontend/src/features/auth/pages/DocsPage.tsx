import { useState } from "react";
import { Link } from "react-router-dom";
import {
  ArrowLeft,
  BookOpen,
  Cpu,
  Shield,
  Zap,
  Terminal as TermIcon,
  FileText,
} from "lucide-react";

export default function DocsPage() {
  const [activeSection, setActiveSection] = useState("overview");

  const sections = [
    { id: "overview", label: "Overview", icon: BookOpen },
    { id: "attention", label: "Attention Mechanisms", icon: Zap },
    { id: "compression", label: "Model Compression", icon: Cpu },
    { id: "runners", label: "Experiment Runners", icon: TermIcon },
  ];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-300 font-sans flex flex-col relative overflow-hidden">
      {/* Background Cyber Grid */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#0f172a_1px,transparent_1px),linear-gradient(to_bottom,#0f172a_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,#000_70%,transparent_100%)] opacity-30 pointer-events-none" />

      {/* Header */}
      <header className="relative z-10 border-b border-slate-900 bg-slate-950/80 backdrop-blur-md px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Link
            to="/"
            className="p-2 hover:bg-white/5 rounded-lg text-slate-400 hover:text-white transition-colors"
          >
            <ArrowLeft size={16} />
          </Link>
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-gradient-to-br from-neon-blue to-purple-600 rounded-lg flex items-center justify-center">
              <img
                src="/logo.jpg"
                alt="NeuralOps"
                className="w-full h-full object-cover rounded-md"
              />
            </div>
            <span className="font-bold text-white tracking-widest text-sm">
              NEURALOPS DOCS
            </span>
          </div>
        </div>
        <Link
          to="/login"
          className="px-4 py-1.5 bg-white/5 hover:bg-white/10 rounded-lg text-xs font-semibold tracking-wider text-white transition-all"
        >
          Console
        </Link>
      </header>

      {/* Workspace */}
      <div className="flex-1 flex relative z-10 max-w-7xl mx-auto w-full px-6 py-8 gap-8">
        {/* Left Sidebar */}
        <aside className="w-64 shrink-0 hidden md:block">
          <nav className="space-y-1 sticky top-24">
            {sections.map((sec) => {
              const Icon = sec.icon;
              return (
                <button
                  key={sec.id}
                  onClick={() => setActiveSection(sec.id)}
                  className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-semibold transition-all ${
                    activeSection === sec.id
                      ? "bg-neon-blue/10 text-neon-blue border border-neon-blue/20"
                      : "text-slate-400 hover:text-slate-200 hover:bg-white/5 border border-transparent"
                  }`}
                >
                  <Icon size={16} />
                  {sec.label}
                </button>
              );
            })}
          </nav>
        </aside>

        {/* Content Pane */}
        <main className="flex-1 max-w-3xl bg-slate-950/40 backdrop-blur-sm border border-slate-900 rounded-2xl p-8 overflow-y-auto leading-relaxed">
          {activeSection === "overview" && (
            <div className="space-y-6 animate-fade-in">
              <h1 className="text-3xl font-black text-white tracking-tight border-b border-slate-900 pb-4">
                Technical Documentation Overview
              </h1>
              <p className="text-slate-400">
                NeuralOps is an end-to-end self-hosted MLOps platform optimized
                for attention-compression research. It lets researchers train,
                evaluate, and monitor compressed Transformer models (LoRA, SVD,
                Pruning) with full local control.
              </p>
              <div className="p-4 rounded-xl bg-neon-blue/5 border border-neon-blue/10 text-sm text-neon-blue flex gap-3">
                <Shield size={20} className="shrink-0" />
                <div>
                  <strong className="font-bold block mb-1">
                    Privacy-First & Self-Hosted
                  </strong>
                  All logs, models, datasets, and visualizations are stored
                  locally on your network or cluster server.
                </div>
              </div>
              <h2 className="text-xl font-bold text-white mt-8">
                Core Features
              </h2>
              <ul className="list-disc pl-5 space-y-2 text-slate-400 text-sm">
                <li>
                  Real-time experiment loss & metric graphs via web sockets.
                </li>
                <li>
                  Dynamic hyperparameter sweep orchestration powered by Optuna.
                </li>
                <li>Overleaf-style shared LaTeX/PDF compiling environment.</li>
                <li>HuggingFace Model Registry and DVC integration.</li>
              </ul>
            </div>
          )}

          {activeSection === "attention" && (
            <div className="space-y-6 animate-fade-in">
              <h1 className="text-3xl font-black text-white tracking-tight border-b border-slate-900 pb-4">
                Optimized Attention Registry
              </h1>
              <p className="text-slate-400">
                The framework supports over 70+ custom attention mechanisms,
                mapped under `ATTENTION_REGISTRY` to speed up Transformer
                sequence processing.
              </p>

              <h2 className="text-xl font-bold text-white mt-8">
                State-of-the-Art Compressions
              </h2>
              <div className="space-y-4">
                <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-900">
                  <h3 className="font-bold text-white mb-1">
                    Aegis Attention (`aegis_v1` - `aegis_v5`)
                  </h3>
                  <p className="text-sm text-slate-400">
                    Security-inspired multi-layer residual compressions for high
                    fidelity.
                  </p>
                </div>
                <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-900">
                  <h3 className="font-bold text-white mb-1">
                    Nova Attention (`nova`, `minimal_nova`)
                  </h3>
                  <p className="text-sm text-slate-400">
                    State-of-the-art compression incorporating ARMOR and AST
                    structural pruning.
                  </p>
                </div>
                <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-900">
                  <h3 className="font-bold text-white mb-1">
                    Aurora Attention (`aurora`, `minimal_aurora`)
                  </h3>
                  <p className="text-sm text-slate-400">
                    High-performance adaptive spectral filtering for linear
                    complex sequencing.
                  </p>
                </div>
              </div>

              <h2 className="text-xl font-bold text-white mt-8">
                Linear & Low-Rank Approximations
              </h2>
              <table className="w-full text-left text-sm border-collapse mt-4">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400">
                    <th className="py-2">Registry Key</th>
                    <th className="py-2">Description</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-900">
                  <tr>
                    <td className="py-3 font-mono text-neon-blue">linformer</td>
                    <td className="py-3 text-slate-400">
                      Fixed-size projection matrix for $O(N)$ efficiency.
                    </td>
                  </tr>
                  <tr>
                    <td className="py-3 font-mono text-neon-blue">svd</td>
                    <td className="py-3 text-slate-400">
                      Singular Value Decomposition low-rank approximation.
                    </td>
                  </tr>
                  <tr>
                    <td className="py-3 font-mono text-neon-blue">performer</td>
                    <td className="py-3 text-slate-400">
                      Positive Orthogonal Random Features (FAVOR+).
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          )}

          {activeSection === "compression" && (
            <div className="space-y-6 animate-fade-in">
              <h1 className="text-3xl font-black text-white tracking-tight border-b border-slate-900 pb-4">
                Model Compressors
              </h1>
              <p className="text-slate-400">
                Optimized post-training quantization, pruning, and low-rank
                mapping modules reside inside the `COMPRESSOR_REGISTRY`.
              </p>

              <h2 className="text-xl font-bold text-white mt-8">
                Matrix Factorization
              </h2>
              <div className="grid grid-cols-1 gap-4">
                <div className="p-4 bg-slate-900/60 border border-slate-900 rounded-xl">
                  <span className="text-xs text-neon-blue font-mono uppercase tracking-wider block mb-1">
                    Swift-SVD
                  </span>
                  <p className="text-sm text-slate-400">
                    Fast, activation-aware Singular Value Decomposition for
                    key-value projections.
                  </p>
                </div>
                <div className="p-4 bg-slate-900/60 border border-slate-900 rounded-xl">
                  <span className="text-xs text-neon-blue font-mono uppercase tracking-wider block mb-1">
                    Wanda & SliceGPT
                  </span>
                  <p className="text-sm text-slate-400">
                    Weight pruning without retraining, utilizing PCA-based
                    projection alignment.
                  </p>
                </div>
                <div className="p-4 bg-slate-900/60 border border-slate-900 rounded-xl">
                  <span className="text-xs text-neon-blue font-mono uppercase tracking-wider block mb-1">
                    ZeroKV & HybridKV
                  </span>
                  <p className="text-sm text-slate-400">
                    Dynamic KV-cache pruning during inference to limit VRAM
                    usage.
                  </p>
                </div>
              </div>
            </div>
          )}

          {activeSection === "runners" && (
            <div className="space-y-6 animate-fade-in">
              <h1 className="text-3xl font-black text-white tracking-tight border-b border-slate-900 pb-4">
                Command Line Experiment Runners
              </h1>
              <p className="text-slate-400">
                Run experiments locally or orchestrate automated sweeps via
                standard Python execution entrypoints.
              </p>

              <h2 className="text-xl font-bold text-white mt-6">
                1. Single Experiment (`run.py`)
              </h2>
              <p className="text-sm text-slate-400">
                Loads a configuration, prepares dataset tokenizers, injects
                custom attention or compressors, and executes training.
              </p>
              <pre className="p-4 bg-slate-900 rounded-xl border border-slate-800 text-xs font-mono text-white overflow-x-auto">
                python run.py --model bert-base-uncased --task mrpc --attn_type
                nova --rank 64 --epochs 3
              </pre>

              <h2 className="text-xl font-bold text-white mt-8">
                2. Batch Sweep (`batch_runner.py`)
              </h2>
              <p className="text-sm text-slate-400">
                Orchestrates matrix tests across all attentions, compressors,
                tasks, and seeds in parallel.
              </p>
              <pre className="p-4 bg-slate-900 rounded-xl border border-slate-800 text-xs font-mono text-white overflow-x-auto">
                python batch_runner.py --epochs 1 --max_samples 1000 --parallel
                4
              </pre>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
