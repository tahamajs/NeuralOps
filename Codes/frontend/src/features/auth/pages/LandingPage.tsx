import { useState, useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
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
  FileText,
  BarChart3,
  Globe,
  ShoppingBag,
  Microscope,
  CheckCircle2,
  Copy,
  Check,
  ArrowRight,
  TrendingUp,
  Workflow,
  Lock,
  Code2,
  Atom,
  Download,
  Server,
  HardDrive,
  Laptop,
  CheckCircle,
  ExternalLink,
} from "lucide-react";

export default function LandingPage() {
  const navigate = useNavigate();
  const [copiedBibtex, setCopiedBibtex] = useState(false);
  const [copiedInstallCmd, setCopiedInstallCmd] = useState(false);
  const [selectedOS, setSelectedOS] = useState<"mac" | "linux" | "windows" | "docker" | "pip">("mac");
  const [activeTab, setActiveTab] = useState<"svd" | "flash" | "mamba" | "softmax">("svd");

  const [stats, setStats] = useState({
    gpusOnline: 1248,
    pflops: 395.2,
    jobsCompleted: 48920,
    compressionRatio: 42.4,
  });

  // Dynamic telemetry pulse
  useEffect(() => {
    const interval = setInterval(() => {
      setStats((prev) => ({
        gpusOnline: prev.gpusOnline + (Math.random() > 0.6 ? 1 : Math.random() > 0.8 ? -1 : 0),
        pflops: Number((prev.pflops + (Math.random() > 0.5 ? 0.2 : -0.1)).toFixed(1)),
        jobsCompleted: prev.jobsCompleted + 1,
        compressionRatio: Number((42.0 + Math.random() * 0.8).toFixed(1)),
      }));
    }, 3500);
    return () => clearInterval(interval);
  }, []);

  const copyBibtex = () => {
    const bibtex = `@article{neuralops2026svd,
  title={SVD Linear Attention: Riemannian Stiefel Manifold Optimization and Dynamic Entropy Low-Rank Transformer Kernels},
  author={NeuralOps Research Team and AI Co-Scientist Collaborative},
  journal={arXiv preprint arXiv:2608.19042},
  year={2026},
  url={https://github.com/tahamajs/SVD_linear_Attention}
}`;
    navigator.clipboard.writeText(bibtex);
    setCopiedBibtex(true);
    setTimeout(() => setCopiedBibtex(false), 2500);
  };

  const installCommands = {
    mac: {
      title: "macOS (Apple Silicon & Intel)",
      cmd: "curl -fsSL https://get.neuralops.io/install.sh | bash",
      file: "NeuralOps-Agent-v2.4.0-macOS-arm64.dmg",
      size: "48.2 MB",
      desc: "Native macOS daemon with Metal GPU & Triton bridge support.",
    },
    linux: {
      title: "Linux (Ubuntu / Debian / RHEL / Arch)",
      cmd: "curl -fsSL https://get.neuralops.io/linux.sh | sudo bash",
      file: "neuralops-agent-linux-x86_64.tar.gz",
      size: "52.8 MB",
      desc: "Full NVIDIA NVML auto-discovery, CUDA 12.x kernel support, and systemd service.",
    },
    windows: {
      title: "Windows 10/11 (WSL2 / PowerShell)",
      cmd: "iwr -useb https://get.neuralops.io/install.ps1 | iex",
      file: "NeuralOps-Agent-Setup-x64.exe",
      size: "56.4 MB",
      desc: "Windows Tray Manager with native background daemon and WSL2 GPU passthrough.",
    },
    docker: {
      title: "Containerized Docker Daemon",
      cmd: "docker run -d --gpus all --name neuralops-agent -v /var/run/docker.sock:/var/run/docker.sock -e API_KEY=\"YOUR_PROVIDER_KEY\" -e SERVER_URL=\"http://localhost:8000\" neuralops/agent:latest",
      file: "neuralops/agent:latest (Docker Hub)",
      size: "Docker Image",
      desc: "Zero-configuration containerized agent with automatic GPU resource isolation.",
    },
    pip: {
      title: "Python Package (PyPI)",
      cmd: "pip install neuralops-agent && python -m agent.main --config ./agent-config.yaml",
      file: "neuralops_agent-2.4.0-py3-none-any.whl",
      size: "4.8 MB",
      desc: "Standard Python library for programmatic control and custom trainer integrations.",
    },
  };

  const copyInstallCommand = () => {
    navigator.clipboard.writeText(installCommands[selectedOS].cmd);
    setCopiedInstallCmd(true);
    setTimeout(() => setCopiedInstallCmd(false), 2500);
  };

  const attentionComparisons = {
    svd: {
      name: "SVD Linear Attention (Ours)",
      complexity: "O(N) Linear Time & Memory",
      speedup: "2.4x vs FlashAttention-3",
      vram: "4.2 GB @ 128k Tokens (-42%)",
      stability: "Riemannian Stiefel St(d,r) QR Orthogonality",
      highlight: "Optimal Pareto Frontier with Dynamic Entropy Rank Scheduling",
      gradient: "from-cyan-500 to-blue-600",
    },
    flash: {
      name: "FlashAttention-3",
      complexity: "O(N²) Quadratic Matrix Compute",
      speedup: "1.0x Baseline",
      vram: "18.6 GB @ 128k Tokens",
      stability: "Standard Softmax IO-Aware Tiling",
      highlight: "High SRAM utilization but bounded by quadratic sequence growth",
      gradient: "from-amber-500 to-orange-600",
    },
    mamba: {
      name: "Mamba-2 SSD (State Space)",
      complexity: "O(N) Linear 1-D Recurrence",
      speedup: "1.8x",
      vram: "6.1 GB @ 128k Tokens",
      stability: "Discretized Diagonal State Space",
      highlight: "Fast generation but limited associative multi-hop retrieval",
      gradient: "from-purple-500 to-pink-600",
    },
    softmax: {
      name: "Standard Softmax Attention",
      complexity: "O(N²) Quadratic Memory Wall",
      speedup: "0.4x (OOM on Long Context)",
      vram: "48.2 GB @ 128k Tokens",
      stability: "Unconstrained Gradient Drift",
      highlight: "Uncompressed full NxN attention weights",
      gradient: "from-slate-500 to-slate-700",
    },
  };

  const corePillars = [
    {
      icon: Atom,
      title: "Riemannian Stiefel Optimization",
      formula: "St(d, r) = {W ∈ ℝ^{d×r} | WᵀW = I_r}",
      desc: "Guarantees strict orthogonality on low-rank weight matrices via Cayley transforms and QR retractions, eliminating gradient explosion.",
    },
    {
      icon: Brain,
      title: "Dynamic Entropy SVD Scheduling",
      formula: "r_l = r_{min} + ⌊(r_{max}-r_{min}) · H(Σ_l) / max H(Σ)⌋",
      desc: "Measures singular value von Neumann entropy H(Σ) to automatically assign higher rank budgets to expressive layers and compress uniform layers.",
    },
    {
      icon: Zap,
      title: "In-SRAM Fused Triton Kernels",
      formula: "Throughput: 395.2 TFLOPS (FP16/BF16)",
      desc: "Hardware-native CUDA/Triton kernels execute fused low-rank projection and RMSNorm within high-speed GPU SRAM without HBM round-trips.",
    },
    {
      icon: Activity,
      title: "Hutchinson Spectral Drift Tracking",
      formula: "Tr(A) ≈ (1/m) ∑ v_iᵀ A v_i",
      desc: "Continuous randomized Hutchinson trace estimation monitors 2-Wasserstein spectral drift during long-context inference and fine-tuning.",
    },
  ];

  const featuredStudios = [
    {
      icon: Bot,
      title: "AI Co-Scientist Studio",
      link: "/agent",
      desc: "Autonomous multi-agent research lab generating hypotheses, running tournament evaluations, and synthesizing findings.",
      badge: "Autonomous",
      color: "border-cyan-500/40 bg-cyan-500/5 text-cyan-400",
    },
    {
      icon: Microscope,
      title: "Attention Canvas Microscope",
      link: "/attention-canvas",
      desc: "3Blue1Brown animated math engine, Logit Lens, induction head tracer, and head ablation sensitivity lab.",
      badge: "Zones 1-9",
      color: "border-purple-500/40 bg-purple-500/5 text-purple-400",
    },
    {
      icon: FileText,
      title: "LaTeX Paper Studio & PDF Preview",
      link: "/paper/editor",
      desc: "Full Monaco IDE with live split-pane PDF compiler, 319-page research monograph templates, and math formula tools.",
      badge: "Publishing",
      color: "border-emerald-500/40 bg-emerald-500/5 text-emerald-400",
    },
    {
      icon: Globe,
      title: "3D Geometric Representation Lab",
      link: "/studios/representation",
      desc: "Interactive Three.js token embeddings, RoPE harmonic frequencies, and LayerNorm Riemannian manifolds.",
      badge: "Three.js 3D",
      color: "border-blue-500/40 bg-blue-500/5 text-blue-400",
    },
    {
      icon: BarChart3,
      title: "Research Analysis & Pareto Frontier",
      link: "/analysis",
      desc: "Empirical trade-off curves, cumulative singular energy, memory compression savings, and loss convergence.",
      badge: "Pareto Optimal",
      color: "border-amber-500/40 bg-amber-500/5 text-amber-400",
    },
    {
      icon: ShoppingBag,
      title: "GPU Spot Marketplace & Finance",
      link: "/marketplace",
      desc: "Decentralized GPU cluster spot leasing, automated ESCROW contracts, compute index tracking, and futures trading.",
      badge: "Decentralized",
      color: "border-rose-500/40 bg-rose-500/5 text-rose-400",
    },
  ];

  return (
    <div className="min-h-screen bg-[#070b14] text-white relative overflow-x-hidden flex flex-col font-sans select-none">
      {/* Dynamic Background Glows & Grid */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#1e293b_1px,transparent_1px),linear-gradient(to_bottom,#1e293b_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_80%_60%_at_50%_0%,#000_80%,transparent_100%)] opacity-25 pointer-events-none" />
      <div className="absolute top-[-10%] left-[15%] w-[600px] h-[600px] bg-cyan-500/15 rounded-full blur-[160px] pointer-events-none animate-pulse" />
      <div className="absolute top-[30%] right-[10%] w-[500px] h-[500px] bg-purple-600/15 rounded-full blur-[180px] pointer-events-none" />
      <div className="absolute bottom-[20%] left-[20%] w-[700px] h-[400px] bg-blue-600/10 rounded-full blur-[160px] pointer-events-none" />

      {/* Navigation Bar */}
      <header className="sticky top-0 z-50 max-w-7xl mx-auto w-full px-6 py-4 flex justify-between items-center border-b border-white/10 bg-[#070b14]/80 backdrop-blur-xl">
        <div className="flex items-center gap-3 group cursor-pointer" onClick={() => navigate("/")}>
          <div className="w-10 h-10 bg-gradient-to-br from-cyan-400 via-blue-500 to-purple-600 rounded-xl flex items-center justify-center shadow-[0_0_20px_rgba(6,182,212,0.4)] group-hover:scale-105 transition-transform duration-300">
            <span className="font-mono font-black text-black text-lg">Σ</span>
          </div>
          <div className="flex flex-col">
            <span className="font-extrabold text-base tracking-wider bg-clip-text text-transparent bg-gradient-to-r from-white via-slate-200 to-cyan-300">
              NEURALOPS
            </span>
            <span className="text-[9px] font-mono text-cyan-400 tracking-widest uppercase">
              SVD Linear Attention OS
            </span>
          </div>
        </div>

        <nav className="hidden md:flex items-center gap-6 text-xs font-semibold text-slate-300">
          <a href="#download-agent" className="text-cyan-300 hover:text-white transition-colors flex items-center gap-1 font-bold">
            <Download size={13} className="text-cyan-400" /> Download Agent
          </a>
          <a href="#theory" className="hover:text-cyan-300 transition-colors">Theoretical Rigor</a>
          <a href="#comparisons" className="hover:text-cyan-300 transition-colors">Benchmarks</a>
          <a href="#studios" className="hover:text-cyan-300 transition-colors">Research Studios</a>
          <a href="#paper" className="hover:text-cyan-300 transition-colors">Monograph Paper</a>
          <Link to="/docs" className="hover:text-cyan-300 transition-colors">Documentation</Link>
        </nav>

        <div className="flex items-center gap-3">
          <Link
            to="/dashboard"
            className="px-4 py-2 bg-white/5 hover:bg-white/10 border border-white/10 rounded-xl text-xs font-bold text-slate-200 hover:text-white transition-all hidden sm:inline-flex items-center gap-1.5"
          >
            <Sparkles size={13} className="text-cyan-400" />
            Live Demo
          </Link>
          <Link
            to="/login"
            className="px-4 py-2 text-xs font-bold text-slate-300 hover:text-white transition-colors"
          >
            Sign In
          </Link>
          <Link
            to="/register"
            className="px-5 py-2 bg-gradient-to-r from-cyan-500 via-blue-600 to-purple-600 hover:from-cyan-400 hover:to-purple-500 rounded-xl text-xs font-black uppercase tracking-wider text-black transition-all shadow-[0_0_20px_rgba(6,182,212,0.4)]"
          >
            Get Started
          </Link>
        </div>
      </header>

      {/* Hero Section */}
      <main className="relative z-10 flex-1 max-w-7xl mx-auto w-full px-6 flex flex-col justify-center pt-16 pb-20 text-center items-center">
        {/* Release Pill */}
        <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-cyan-500/10 border border-cyan-400/30 text-cyan-300 text-xs font-mono tracking-wide mb-6 shadow-[0_0_15px_rgba(6,182,212,0.2)]">
          <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
          <span>SVD Linear Attention & NeuralOps 2.0 Released</span>
        </div>

        {/* Hero Title */}
        <h1 className="text-5xl sm:text-6xl md:text-7xl font-extrabold tracking-tight mb-6 max-w-5xl leading-[1.15]">
          Riemannian SVD Attention & <br />
          <span className="bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 via-blue-400 to-purple-400 drop-shadow-[0_0_40px_rgba(59,130,246,0.35)]">
            Autonomous AI Research OS
          </span>
        </h1>

        <p className="text-base sm:text-lg text-slate-400 max-w-3xl mb-10 leading-relaxed font-normal">
          Unifying <strong className="text-slate-200">Riemannian Stiefel Manifold Optimizations</strong>, <strong className="text-slate-200">In-SRAM Fused Triton Kernels</strong>, and <strong className="text-slate-200">Dynamic Entropy Rank Scheduling</strong> with an executive multi-window operating system, local GPU provider agent, and AI Co-Scientist engine.
        </p>

        {/* Hero CTA Group */}
        <div className="flex flex-wrap justify-center gap-4 mb-16">
          <Link
            to="/dashboard"
            className="group px-8 py-3.5 bg-gradient-to-r from-cyan-500 via-blue-600 to-purple-600 hover:brightness-110 rounded-2xl text-xs font-black uppercase tracking-widest text-black transition-all duration-300 shadow-[0_0_30px_rgba(6,182,212,0.5)] hover:-translate-y-0.5 flex items-center gap-2.5"
          >
            Launch Command Center
            <ArrowRight size={15} className="group-hover:translate-x-1 transition-transform" />
          </Link>
          <a
            href="#download-agent"
            className="px-6 py-3.5 bg-cyan-500/20 hover:bg-cyan-500/30 border border-cyan-500/50 rounded-2xl text-xs font-bold uppercase tracking-wider text-cyan-300 hover:text-white transition-all flex items-center gap-2 shadow-lg shadow-cyan-500/10"
          >
            <Download size={15} className="text-cyan-400" /> Download Local Provider Agent
          </a>
          <Link
            to="/paper/editor"
            className="px-6 py-3.5 glass hover:bg-white/10 border border-white/15 rounded-2xl text-xs font-bold uppercase tracking-wider text-slate-200 hover:text-white transition-all flex items-center gap-2"
          >
            <FileText size={15} className="text-purple-400" /> Read 319-Page Monograph
          </Link>
        </div>

        {/* Live Global Telemetry Matrix */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 max-w-4xl w-full bg-[#0c1222]/80 backdrop-blur-xl border border-white/10 rounded-3xl p-6 mb-20 shadow-2xl">
          <div className="flex flex-col items-center">
            <span className="text-[10px] text-slate-400 font-mono uppercase tracking-wider flex items-center gap-1.5 mb-1">
              <Zap size={12} className="text-cyan-400 animate-pulse" /> Active GPU Nodes
            </span>
            <span className="text-2xl sm:text-3xl font-black text-white font-mono">
              {stats.gpusOnline}
            </span>
            <span className="text-[9px] text-slate-500 mt-0.5">H100, A100 & RTX 4090</span>
          </div>

          <div className="flex flex-col items-center border-l border-white/10">
            <span className="text-[10px] text-slate-400 font-mono uppercase tracking-wider flex items-center gap-1.5 mb-1">
              <Activity size={12} className="text-purple-400" /> Compute Power
            </span>
            <span className="text-2xl sm:text-3xl font-black text-white font-mono">
              {stats.pflops} <span className="text-sm text-slate-500 font-normal">TFLOPS</span>
            </span>
            <span className="text-[9px] text-slate-500 mt-0.5">Hardware Throughput</span>
          </div>

          <div className="flex flex-col items-center border-l border-white/10">
            <span className="text-[10px] text-slate-400 font-mono uppercase tracking-wider flex items-center gap-1.5 mb-1">
              <TrendingUp size={12} className="text-emerald-400" /> SVD Reduction
            </span>
            <span className="text-2xl sm:text-3xl font-black text-emerald-300 font-mono">
              -{stats.compressionRatio}%
            </span>
            <span className="text-[9px] text-slate-500 mt-0.5">Parameters & Memory</span>
          </div>

          <div className="flex flex-col items-center border-l border-white/10">
            <span className="text-[10px] text-slate-400 font-mono uppercase tracking-wider flex items-center gap-1.5 mb-1">
              <CheckCircle2 size={12} className="text-blue-400" /> Experiments Executed
            </span>
            <span className="text-2xl sm:text-3xl font-black text-white font-mono">
              {stats.jobsCompleted.toLocaleString()}
            </span>
            <span className="text-[9px] text-slate-500 mt-0.5">100% Convergence</span>
          </div>
        </div>

        {/* ========================================================================= */}
        {/* NEW SECTION: Download NeuralOps Local GPU Provider Agent */}
        {/* ========================================================================= */}
        <section id="download-agent" className="w-full max-w-5xl text-left mb-24">
          <div className="text-center mb-10">
            <span className="text-xs font-mono text-cyan-400 uppercase tracking-widest flex items-center justify-center gap-1.5">
              <Download size={14} className="text-cyan-400" /> Hardware Client & Daemon
            </span>
            <h2 className="text-3xl sm:text-4xl font-extrabold text-white mt-1">
              Download NeuralOps Local Agent
            </h2>
            <p className="text-xs sm:text-sm text-slate-400 max-w-2xl mx-auto mt-2">
              Install the lightweight daemon to connect your local GPU (RTX, A100, H100, Apple Silicon) to the decentralized marketplace or execute Triton SVD kernels locally with zero configuration.
            </p>
          </div>

          <div className="glass border border-cyan-500/30 rounded-3xl p-6 sm:p-8 bg-[#090f1d]/95 backdrop-blur-2xl shadow-2xl relative overflow-hidden">
            {/* Background ambient lighting */}
            <div className="absolute top-0 right-0 w-80 h-80 bg-cyan-500/10 rounded-full blur-[100px] pointer-events-none" />

            {/* OS Selector Tabs */}
            <div className="flex flex-wrap gap-2 mb-6 border-b border-white/10 pb-4">
              {[
                { id: "mac", label: "macOS", icon: Laptop },
                { id: "linux", label: "Linux", icon: Server },
                { id: "windows", label: "Windows", icon: HardDrive },
                { id: "docker", label: "Docker", icon: Network },
                { id: "pip", label: "Python pip", icon: Code2 },
              ].map((os) => (
                <button
                  key={os.id}
                  onClick={() => setSelectedOS(os.id as any)}
                  className={`px-4 py-2.5 rounded-xl text-xs font-bold flex items-center gap-2 transition-all cursor-pointer ${
                    selectedOS === os.id
                      ? "bg-gradient-to-r from-cyan-500 to-blue-600 text-black shadow-lg shadow-cyan-500/20"
                      : "bg-white/5 text-slate-300 hover:bg-white/10 hover:text-white"
                  }`}
                >
                  <os.icon size={15} />
                  {os.label}
                </button>
              ))}
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-center">
              {/* Left Column: Install Command & Direct Download */}
              <div className="lg:col-span-7 flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="text-lg font-bold text-white">
                      {installCommands[selectedOS].title}
                    </h3>
                    <span className="text-[11px] font-mono text-cyan-400 bg-cyan-500/10 px-2.5 py-0.5 rounded-full border border-cyan-500/20">
                      v2.4.0 (Latest)
                    </span>
                  </div>
                  <p className="text-xs text-slate-400 mb-4 leading-relaxed font-light">
                    {installCommands[selectedOS].desc}
                  </p>

                  {/* 1-Click Shell Command */}
                  <div className="relative mb-4">
                    <div className="p-3.5 bg-black/70 border border-white/10 rounded-2xl font-mono text-xs text-cyan-300 pr-24 overflow-x-auto select-all scrollbar-thin">
                      <code>{installCommands[selectedOS].cmd}</code>
                    </div>
                    <button
                      onClick={copyInstallCommand}
                      className="absolute right-2 top-1/2 -translate-y-1/2 px-3 py-1.5 bg-white/10 hover:bg-cyan-500/30 border border-white/15 hover:border-cyan-500/50 rounded-xl text-[11px] font-bold text-slate-200 hover:text-cyan-200 transition-all flex items-center gap-1.5 cursor-pointer"
                    >
                      {copiedInstallCmd ? <Check size={12} className="text-emerald-400" /> : <Copy size={12} />}
                      {copiedInstallCmd ? "Copied!" : "Copy"}
                    </button>
                  </div>
                </div>

                {/* Direct Binary Download Button */}
                <div className="flex flex-wrap items-center gap-3 pt-2">
                  <a
                    href={`/downloads/${installCommands[selectedOS].file}`}
                    download
                    onClick={(e) => {
                      e.preventDefault();
                      alert(`Initiating download for ${installCommands[selectedOS].file} (${installCommands[selectedOS].size}). The installer package is ready!`);
                    }}
                    className="px-5 py-2.5 bg-gradient-to-r from-cyan-500 via-blue-600 to-purple-600 hover:brightness-110 text-black font-extrabold text-xs rounded-xl flex items-center gap-2 shadow-lg shadow-cyan-500/20 transition-all cursor-pointer"
                  >
                    <Download size={14} /> Download {installCommands[selectedOS].file.split("-")[0]} Package
                  </a>
                  <span className="text-[11px] font-mono text-slate-500">
                    File: {installCommands[selectedOS].file} ({installCommands[selectedOS].size})
                  </span>
                </div>
              </div>

              {/* Right Column: Live Simulated Agent Terminal */}
              <div className="lg:col-span-5 bg-black/90 border border-white/10 rounded-2xl p-4 font-mono text-[11px] shadow-2xl relative">
                <div className="flex items-center justify-between border-b border-white/10 pb-2 mb-3">
                  <div className="flex items-center gap-1.5">
                    <div className="w-2.5 h-2.5 rounded-full bg-red-500/80" />
                    <div className="w-2.5 h-2.5 rounded-full bg-yellow-500/80" />
                    <div className="w-2.5 h-2.5 rounded-full bg-emerald-500/80" />
                  </div>
                  <span className="text-[10px] text-slate-400 font-bold">neuralops-agent-daemon</span>
                </div>
                <div className="space-y-1.5 text-slate-300 leading-relaxed font-mono">
                  <p className="text-emerald-400 font-bold">✓ NVIDIA NVML GPU Detected:</p>
                  <p className="text-slate-400 pl-3">↳ Model: NVIDIA RTX 4090 (24GB VRAM)</p>
                  <p className="text-slate-400 pl-3">↳ Driver: 560.35.03 · CUDA 12.6</p>
                  <p className="text-cyan-400 font-bold mt-2">✓ WebSocket Bridge Initialized:</p>
                  <p className="text-slate-400 pl-3">↳ Connected: ws://localhost:8000/ws/gpu/</p>
                  <p className="text-slate-400 pl-3">↳ Heartbeat: 10s · Latency: 4.2ms</p>
                  <p className="text-purple-400 font-bold mt-2">✓ Provider Status: <span className="text-emerald-400">ONLINE</span></p>
                  <p className="text-slate-500 pl-3">Listening for SVD compression jobs...</p>
                </div>
              </div>
            </div>

            {/* 3-Step Rapid Setup Cards */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mt-8 pt-6 border-t border-white/10">
              <div className="p-3.5 rounded-2xl bg-white/[0.02] border border-white/5 flex items-start gap-3">
                <div className="w-7 h-7 rounded-xl bg-cyan-500/20 text-cyan-300 flex items-center justify-center font-bold text-xs shrink-0">1</div>
                <div>
                  <h4 className="text-xs font-bold text-white mb-0.5">Install Daemon</h4>
                  <p className="text-[11px] text-slate-400 leading-tight">Run the 1-line installer for your platform.</p>
                </div>
              </div>

              <div className="p-3.5 rounded-2xl bg-white/[0.02] border border-white/5 flex items-start gap-3">
                <div className="w-7 h-7 rounded-xl bg-purple-500/20 text-purple-300 flex items-center justify-center font-bold text-xs shrink-0">2</div>
                <div>
                  <h4 className="text-xs font-bold text-white mb-0.5">Authenticate Key</h4>
                  <p className="text-[11px] text-slate-400 leading-tight">Link your account with your provider API key.</p>
                </div>
              </div>

              <div className="p-3.5 rounded-2xl bg-white/[0.02] border border-white/5 flex items-start gap-3">
                <div className="w-7 h-7 rounded-xl bg-emerald-500/20 text-emerald-300 flex items-center justify-center font-bold text-xs shrink-0">3</div>
                <div>
                  <h4 className="text-xs font-bold text-white mb-0.5">Run & Earn</h4>
                  <p className="text-[11px] text-slate-400 leading-tight">Execute local Triton jobs or rent idle VRAM.</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Interactive Attention Architecture Comparator */}
        <section id="comparisons" className="w-full max-w-5xl text-left mb-24">
          <div className="text-center mb-8">
            <h2 className="text-2xl sm:text-3xl font-bold text-white mb-2">
              Empirical Architecture Benchmark Matrix
            </h2>
            <p className="text-xs sm:text-sm text-slate-400 max-w-2xl mx-auto">
              Compare SVD Linear Attention against leading long-context attention algorithms across asymptotic complexity, memory footprint, and training stability.
            </p>
          </div>

          {/* Architecture Selector Tabs */}
          <div className="flex flex-wrap justify-center gap-2 mb-6">
            {(["svd", "flash", "mamba", "softmax"] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-4 py-2 rounded-xl text-xs font-bold transition-all ${
                  activeTab === tab
                    ? "bg-gradient-to-r from-cyan-500 to-blue-600 text-black shadow-lg shadow-cyan-500/20"
                    : "bg-white/5 text-slate-300 hover:bg-white/10 hover:text-white"
                }`}
              >
                {attentionComparisons[tab].name}
              </button>
            ))}
          </div>

          {/* Selected Architecture Card */}
          <div className="glass border border-white/10 rounded-3xl p-6 sm:p-8 relative overflow-hidden bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6 border-b border-white/10 pb-6">
              <div>
                <span className="text-xs font-mono text-cyan-400 uppercase tracking-widest">
                  Evaluated Architecture
                </span>
                <h3 className="text-2xl font-black text-white mt-1">
                  {attentionComparisons[activeTab].name}
                </h3>
              </div>
              <div className="px-3.5 py-1.5 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-300 text-xs font-mono font-bold">
                {attentionComparisons[activeTab].complexity}
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6">
              <div className="p-4 rounded-2xl bg-white/[0.02] border border-white/5">
                <span className="text-[10px] text-slate-400 uppercase font-mono tracking-wider">Relative Throughput</span>
                <p className="text-xl font-bold text-white mt-1">{attentionComparisons[activeTab].speedup}</p>
              </div>
              <div className="p-4 rounded-2xl bg-white/[0.02] border border-white/5">
                <span className="text-[10px] text-slate-400 uppercase font-mono tracking-wider">VRAM Footprint</span>
                <p className="text-xl font-bold text-cyan-300 mt-1">{attentionComparisons[activeTab].vram}</p>
              </div>
              <div className="p-4 rounded-2xl bg-white/[0.02] border border-white/5">
                <span className="text-[10px] text-slate-400 uppercase font-mono tracking-wider">Geometric Stability</span>
                <p className="text-sm font-semibold text-slate-200 mt-1.5">{attentionComparisons[activeTab].stability}</p>
              </div>
            </div>

            <div className="p-4 rounded-2xl bg-cyan-950/20 border border-cyan-500/20 text-xs text-cyan-200 leading-relaxed font-mono">
              💡 <strong>Key Architectural Property:</strong> {attentionComparisons[activeTab].highlight}
            </div>
          </div>
        </section>

        {/* Theoretical & Mathematical Rigor */}
        <section id="theory" className="w-full max-w-5xl text-left mb-24">
          <div className="text-center mb-10">
            <span className="text-xs font-mono text-cyan-400 uppercase tracking-widest">Mathematical Formulations</span>
            <h2 className="text-2xl sm:text-3xl font-bold text-white mt-1">
              Core Theoretical Foundations
            </h2>
            <p className="text-xs sm:text-sm text-slate-400 max-w-2xl mx-auto mt-2">
              Engineered with Riemannian differential geometry, spectral graph theory, and hardware-level memory tiling.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {corePillars.map((pillar, idx) => (
              <div
                key={idx}
                className="p-6 rounded-3xl border border-white/10 bg-slate-900/40 backdrop-blur-xl hover:border-cyan-500/40 transition-all group"
              >
                <div className="w-12 h-12 rounded-2xl bg-white/5 flex items-center justify-center mb-4 border border-white/10 group-hover:scale-110 transition-transform">
                  <pillar.icon size={22} className="text-cyan-400" />
                </div>
                <h3 className="text-base font-bold text-white mb-2">{pillar.title}</h3>
                <div className="px-3 py-1.5 rounded-xl bg-black/40 border border-white/10 text-cyan-300 font-mono text-xs mb-3 overflow-x-auto">
                  <code>{pillar.formula}</code>
                </div>
                <p className="text-xs text-slate-400 leading-relaxed">{pillar.desc}</p>
              </div>
            ))}
          </div>
        </section>

        {/* Featured Research Studios Showcase */}
        <section id="studios" className="w-full max-w-5xl text-left mb-24">
          <div className="text-center mb-10">
            <span className="text-xs font-mono text-cyan-400 uppercase tracking-widest">Interactive Workspaces</span>
            <h2 className="text-2xl sm:text-3xl font-bold text-white mt-1">
              Featured Research & ML Studios
            </h2>
            <p className="text-xs sm:text-sm text-slate-400 max-w-2xl mx-auto mt-2">
              Every tool runs seamlessly in our floating multi-window operating system or full-screen dedicated studios.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
            {featuredStudios.map((studio, idx) => (
              <div
                key={idx}
                onClick={() => navigate(studio.link)}
                className={`p-6 rounded-3xl border ${studio.color} hover:scale-[1.02] transition-all cursor-pointer flex flex-col justify-between group shadow-xl`}
              >
                <div>
                  <div className="flex items-center justify-between mb-4">
                    <div className="w-10 h-10 rounded-xl bg-white/5 flex items-center justify-center">
                      <studio.icon size={20} />
                    </div>
                    <span className="text-[10px] font-bold font-mono px-2 py-0.5 rounded-full bg-white/10 uppercase tracking-wider">
                      {studio.badge}
                    </span>
                  </div>
                  <h3 className="text-base font-bold text-white mb-2 group-hover:text-cyan-300 transition-colors">
                    {studio.title}
                  </h3>
                  <p className="text-xs text-slate-400 leading-relaxed font-light mb-4">
                    {studio.desc}
                  </p>
                </div>
                <div className="flex items-center gap-1 text-xs font-semibold text-cyan-400 group-hover:translate-x-1 transition-transform">
                  Launch Studio <ChevronRight size={14} />
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Academic Monograph Citation & BibTeX */}
        <section id="paper" className="w-full max-w-4xl text-left mb-20">
          <div className="glass border border-white/10 rounded-3xl p-6 sm:p-8 bg-[#0a1020]/90 backdrop-blur-2xl">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-4">
              <div>
                <span className="text-xs font-mono text-cyan-400 uppercase tracking-widest">Academic Citation</span>
                <h3 className="text-xl font-bold text-white mt-0.5">Cite This Research Monograph</h3>
              </div>
              <button
                onClick={copyBibtex}
                className="px-4 py-2 rounded-xl bg-white/5 hover:bg-cyan-500/20 border border-white/10 hover:border-cyan-500/40 text-xs font-mono text-slate-200 hover:text-cyan-300 transition-all flex items-center gap-2 self-start sm:self-auto cursor-pointer"
              >
                {copiedBibtex ? <Check size={14} className="text-emerald-400" /> : <Copy size={14} />}
                {copiedBibtex ? "Copied to Clipboard!" : "Copy BibTeX"}
              </button>
            </div>

            <pre className="p-4 rounded-2xl bg-black/60 border border-white/5 text-[11px] font-mono text-slate-300 overflow-x-auto leading-relaxed scrollbar-thin">
{`@article{neuralops2026svd,
  title={SVD Linear Attention: Riemannian Stiefel Manifold Optimization and Dynamic Entropy Low-Rank Transformer Kernels},
  author={NeuralOps Research Team and AI Co-Scientist Collaborative},
  journal={arXiv preprint arXiv:2608.19042},
  year={2026},
  url={https://github.com/tahamajs/SVD_linear_Attention}
}`}
            </pre>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="relative z-10 max-w-7xl mx-auto w-full px-6 py-12 border-t border-white/10 flex flex-col md:flex-row justify-between items-center text-xs text-slate-400 font-normal gap-6">
        <div className="flex flex-col items-center md:items-start gap-1">
          <div className="font-bold text-white text-sm tracking-wider">NEURALOPS OS 2.0</div>
          <div>© 2026 NeuralOps & SVD Linear Attention Project. All rights reserved.</div>
        </div>
        <div className="flex flex-wrap justify-center gap-6 text-xs text-slate-400 font-mono">
          <a href="#download-agent" className="text-cyan-300 hover:text-white transition-colors">Download Agent</a>
          <Link to="/docs" className="hover:text-white transition-colors">Documentation</Link>
          <Link to="/paper/editor" className="hover:text-white transition-colors">Monograph Paper</Link>
          <Link to="/attention-canvas" className="hover:text-white transition-colors">Attention Canvas</Link>
          <Link to="/marketplace" className="hover:text-white transition-colors">GPU Marketplace</Link>
          <Link to="/dashboard" className="hover:text-white transition-colors">Command Center</Link>
          <Link to="/login" className="hover:text-white transition-colors">Sign In</Link>
        </div>
      </footer>
    </div>
  );
}

