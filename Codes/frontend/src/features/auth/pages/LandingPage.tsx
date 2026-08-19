import { useState, useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuthStore } from "../store/authStore";
import {
  Brain,
  Bot,
  Server,
  Layers,
  FileText,
  Activity,
  ShieldCheck,
  Zap,
  Terminal,
  ArrowRight,
  Lock,
} from "lucide-react";

export default function LandingPage() {
  const navigate = useNavigate();
  const isAuthenticated = useAuthStore((state) => state.isAuthenticated);

  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    if (isAuthenticated) {
      navigate("/dashboard");
    }
  }, [isAuthenticated, navigate]);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 50);
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const { currentTarget, clientX, clientY } = e;
    const { left, top } = currentTarget.getBoundingClientRect();
    setMousePosition({ x: clientX - left, y: clientY - top });
  };

  return (
    <div 
      className="min-h-screen bg-[#030712] text-slate-200 font-sans overflow-x-hidden selection:bg-cyan-500/30"
      onMouseMove={handleMouseMove}
    >
      {/* Dynamic Cursor Glow */}
      <div 
        className="pointer-events-none fixed inset-0 z-50 transition-opacity duration-300"
        style={{
          background: `radial-gradient(600px circle at ${mousePosition.x}px ${mousePosition.y}px, rgba(6,182,212,0.03), transparent 40%)`
        }}
      />

      {/* Grid Background */}
      <div className="fixed inset-0 z-0 opacity-20 pointer-events-none">
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px]"></div>
        <div className="absolute left-0 right-0 top-0 -z-10 m-auto h-[310px] w-[310px] rounded-full bg-cyan-500 opacity-20 blur-[100px]"></div>
      </div>

      {/* Navbar */}
      <nav className={`fixed top-0 w-full z-50 transition-all duration-300 border-b ${scrolled ? "bg-[#030712]/80 backdrop-blur-md border-white/10 py-3" : "bg-transparent border-transparent py-5"}`}>
        <div className="max-w-7xl mx-auto px-6 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-400 to-blue-600 flex items-center justify-center font-black text-black shadow-[0_0_15px_rgba(6,182,212,0.5)]">
              Σ
            </div>
            <span className="font-extrabold text-lg tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white to-slate-400">
              NeuralOps
            </span>
          </div>
          
          <div className="hidden md:flex items-center gap-8 text-sm font-medium text-slate-400">
            <a href="#features" className="hover:text-cyan-400 transition-colors">Features</a>
            <a href="#architecture" className="hover:text-cyan-400 transition-colors">Architecture</a>
            <a href="#marketplace" className="hover:text-cyan-400 transition-colors">Marketplace</a>
            <Link to="/docs" className="hover:text-cyan-400 transition-colors">Docs</Link>
          </div>

          <div className="flex items-center gap-4">
            <Link to="/login" className="text-sm font-bold text-slate-300 hover:text-white transition-colors">
              Sign In
            </Link>
            <Link to="/register" className="text-sm font-bold px-5 py-2 rounded-full bg-white text-black hover:bg-cyan-50 hover:scale-105 transition-all shadow-[0_0_20px_rgba(255,255,255,0.2)]">
              Get Started
            </Link>
          </div>
        </div>
      </nav>

      <main className="relative z-10">
        {/* Hero Section */}
        <section className="relative pt-40 pb-20 px-6 max-w-7xl mx-auto flex flex-col items-center text-center">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 text-xs font-mono font-bold tracking-wide mb-8 animate-fade-in-up">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-cyan-500"></span>
            </span>
            NeuralOps OS 2.0 Released
          </div>

          <h1 className="text-6xl md:text-8xl font-black tracking-tighter mb-6 leading-tight max-w-5xl">
            The Operating System for <br />
            <span className="bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-500">
              Agentic Intelligence
            </span>
          </h1>

          <p className="text-lg md:text-xl text-slate-400 max-w-3xl mb-12 leading-relaxed font-light">
            Unifying Riemannian Stiefel Manifold Optimization, O(N) SVD Linear Attention, and an autonomous decentralized GPU marketplace into a single, cohesive research platform.
          </p>

          <div className="flex flex-col sm:flex-row items-center gap-4 w-full justify-center">
            <Link to="/register" className="w-full sm:w-auto px-8 py-4 rounded-full bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-black font-extrabold text-sm uppercase tracking-wider flex items-center justify-center gap-2 transition-all hover:shadow-[0_0_30px_rgba(6,182,212,0.4)] hover:-translate-y-1">
              Launch Dashboard <ArrowRight size={18} />
            </Link>
            <a href="#install" className="w-full sm:w-auto px-8 py-4 rounded-full bg-white/5 hover:bg-white/10 border border-white/10 text-white font-bold text-sm uppercase tracking-wider flex items-center justify-center gap-2 transition-all">
              <Terminal size={18} className="text-cyan-400" /> CLI Install
            </a>
          </div>

          {/* Abstract Dashboard Preview */}
          <div className="mt-24 w-full max-w-5xl relative rounded-2xl border border-white/10 bg-[#0a0f1c]/80 backdrop-blur-xl shadow-2xl p-2 overflow-hidden group">
            <div className="absolute top-0 inset-x-0 h-px bg-gradient-to-r from-transparent via-cyan-500 to-transparent opacity-50"></div>
            <div className="rounded-xl border border-white/5 bg-black/50 aspect-[16/9] flex flex-col">
              {/* Fake Window Header */}
              <div className="h-10 border-b border-white/5 flex items-center px-4 gap-2">
                <div className="flex gap-1.5">
                  <div className="w-3 h-3 rounded-full bg-red-500/80"></div>
                  <div className="w-3 h-3 rounded-full bg-amber-500/80"></div>
                  <div className="w-3 h-3 rounded-full bg-emerald-500/80"></div>
                </div>
                <div className="mx-auto text-xs font-mono text-slate-500 flex items-center gap-2">
                  <Lock size={10} /> 127.0.0.1:3000/dashboard
                </div>
              </div>
              {/* Fake UI Content */}
              <div className="flex-1 p-6 grid grid-cols-3 gap-6 opacity-80 group-hover:opacity-100 transition-opacity duration-700">
                <div className="col-span-2 space-y-4">
                  <div className="h-32 rounded-lg bg-gradient-to-br from-cyan-500/10 to-blue-600/10 border border-cyan-500/20 flex items-end p-4">
                     <div className="w-full h-1/2 flex items-end gap-1">
                       {[40,70,45,90,65,80,100,55,75,85].map((h, i) => (
                         <div key={i} className="flex-1 bg-cyan-400/50 rounded-t-sm" style={{ height: `${h}%` }}></div>
                       ))}
                     </div>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="h-24 rounded-lg bg-white/5 border border-white/5 p-4">
                      <div className="h-4 w-1/3 bg-slate-700 rounded mb-4"></div>
                      <div className="h-8 w-1/2 bg-white/20 rounded"></div>
                    </div>
                    <div className="h-24 rounded-lg bg-white/5 border border-white/5 p-4">
                      <div className="h-4 w-1/3 bg-slate-700 rounded mb-4"></div>
                      <div className="h-8 w-1/2 bg-white/20 rounded"></div>
                    </div>
                  </div>
                </div>
                <div className="space-y-4">
                  <div className="h-full rounded-lg bg-white/5 border border-white/5 p-4 flex flex-col gap-3">
                    <div className="h-4 w-1/2 bg-slate-700 rounded mb-2"></div>
                    {[1,2,3,4,5].map(i => (
                      <div key={i} className="h-10 bg-white/5 rounded flex items-center px-3 gap-3">
                         <div className="w-6 h-6 rounded-full bg-cyan-500/20"></div>
                         <div className="h-2 w-full bg-slate-600 rounded"></div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Feature Highlights Grid */}
        <section id="features" className="py-24 relative">
          <div className="max-w-7xl mx-auto px-6">
            <div className="text-center mb-16">
              <h2 className="text-3xl md:text-5xl font-bold mb-4">Everything You Need to Build <br/><span className="text-cyan-400">Next-Gen Models</span></h2>
              <p className="text-slate-400 text-lg">A fully integrated suite combining research, engineering, and deployment.</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {[
                { icon: Brain, title: "SVD Linear Attention", desc: "O(N) complexity with mathematically proven Riemannian Stiefel manifold optimization.", color: "text-purple-400", bg: "bg-purple-500/10" },
                { icon: Bot, title: "AI Co-Scientist", desc: "Autonomous multi-agent swarms that research, hypothesize, and write LaTeX papers.", color: "text-cyan-400", bg: "bg-cyan-500/10" },
                { icon: Server, title: "Decentralized GPU Net", desc: "Spot market for idle VRAM. Rent or provide H100s, A100s, or local RTX nodes.", color: "text-emerald-400", bg: "bg-emerald-500/10" },
                { icon: Layers, title: "Workspace Studio", desc: "Visual canvas for transformer interpretability, logit lenses, and ablation studies.", color: "text-blue-400", bg: "bg-blue-500/10" },
                { icon: FileText, title: "Live Paper Editor", desc: "Real-time pdflatex compilation embedded in a Monaco editor with live streaming logs.", color: "text-amber-400", bg: "bg-amber-500/10" },
                { icon: Activity, title: "Pre-Flight Validator", desc: "Simulate and validate distributed training jobs and escrow limits before dispatch.", color: "text-rose-400", bg: "bg-rose-500/10" },
              ].map((f, i) => (
                <div key={i} className="group p-8 rounded-3xl bg-white/[0.02] border border-white/5 hover:bg-white/[0.04] hover:border-white/10 transition-all duration-300 relative overflow-hidden">
                  <div className={`w-12 h-12 rounded-xl ${f.bg} flex items-center justify-center mb-6`}>
                    <f.icon size={24} className={f.color} />
                  </div>
                  <h3 className="text-xl font-bold mb-3">{f.title}</h3>
                  <p className="text-sm text-slate-400 leading-relaxed">{f.desc}</p>
                  
                  {/* Subtle hover gradient */}
                  <div className="absolute inset-0 bg-gradient-to-br from-transparent to-white/5 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none"></div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* The Paradigm Shift (Architecture) */}
        <section id="architecture" className="py-24 bg-[#0a0f1c] relative border-y border-white/5">
          <div className="max-w-7xl mx-auto px-6 grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
            <div>
              <div className="inline-block px-3 py-1 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400 text-xs font-mono font-bold tracking-wide mb-6">
                ARCHITECTURE
              </div>
              <h2 className="text-4xl md:text-5xl font-bold mb-6">
                Breaking the <br/> <span className="text-cyan-400">Memory Wall</span>
              </h2>
              <p className="text-slate-400 mb-8 leading-relaxed">
                Standard Softmax Attention scales quadratically O(N²), severely limiting long-context capabilities. Our novel SVD Linear Attention leverages low-rank parameterization and dynamic entropy scheduling to achieve true O(N) scaling.
              </p>
              
              <div className="space-y-6">
                <div className="flex items-start gap-4">
                  <div className="w-10 h-10 rounded-full bg-emerald-500/10 flex items-center justify-center shrink-0 mt-1">
                    <ShieldCheck size={20} className="text-emerald-400" />
                  </div>
                  <div>
                    <h4 className="font-bold text-white mb-1">Stiefel Manifold Preservation</h4>
                    <p className="text-sm text-slate-400">QR retractions ensure absolute orthogonality during gradient updates, preventing explosive divergence.</p>
                  </div>
                </div>
                <div className="flex items-start gap-4">
                  <div className="w-10 h-10 rounded-full bg-cyan-500/10 flex items-center justify-center shrink-0 mt-1">
                    <Zap size={20} className="text-cyan-400" />
                  </div>
                  <div>
                    <h4 className="font-bold text-white mb-1">In-SRAM Triton Fusion</h4>
                    <p className="text-sm text-slate-400">Custom GPU kernels keep matrices entirely within SRAM, completely eliminating HBM round-trips.</p>
                  </div>
                </div>
              </div>
            </div>
            
            {/* Visualizer */}
            <div className="relative aspect-square md:aspect-[4/3] w-full bg-black/40 rounded-3xl border border-white/10 p-8 flex flex-col justify-center overflow-hidden">
               <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[200%] h-full bg-[radial-gradient(ellipse_at_top,rgba(6,182,212,0.1),transparent_50%)]"></div>
               
               <div className="relative z-10 flex flex-col gap-8">
                 {/* O(N^2) Softmax */}
                 <div>
                   <div className="flex justify-between text-xs font-mono mb-2">
                     <span className="text-slate-400">Softmax Attention</span>
                     <span className="text-rose-400 font-bold">O(N²)</span>
                   </div>
                   <div className="h-8 w-full bg-rose-500/10 rounded-lg overflow-hidden flex">
                     <div className="h-full bg-rose-500 w-full animate-[grow_3s_ease-in-out_infinite_alternate]" style={{ transformOrigin: 'left' }}></div>
                   </div>
                   <div className="flex justify-between mt-1 text-[10px] text-slate-500">
                     <span>4K Tokens</span>
                     <span>128K Tokens (OOM)</span>
                   </div>
                 </div>

                 {/* O(N) SVD */}
                 <div>
                   <div className="flex justify-between text-xs font-mono mb-2">
                     <span className="text-cyan-400">NeuralOps SVD Linear</span>
                     <span className="text-cyan-400 font-bold">O(N)</span>
                   </div>
                   <div className="h-8 w-full bg-cyan-500/10 rounded-lg overflow-hidden flex">
                     <div className="h-full bg-cyan-400 w-1/4"></div>
                   </div>
                   <div className="flex justify-between mt-1 text-[10px] text-slate-500">
                     <span>4K Tokens</span>
                     <span>128K Tokens (Constant Memory)</span>
                   </div>
                 </div>
               </div>
            </div>
          </div>
        </section>

        {/* CLI Install Section */}
        <section id="install" className="py-24 relative">
          <div className="max-w-4xl mx-auto px-6 text-center">
            <h2 className="text-3xl font-bold mb-6">Join the Network in Seconds</h2>
            <p className="text-slate-400 mb-10">Run our lightweight rust-based daemon to connect your local GPU to the platform.</p>
            
            <div className="bg-[#050505] rounded-2xl border border-white/10 p-4 shadow-2xl relative overflow-hidden text-left max-w-2xl mx-auto">
              <div className="flex items-center gap-2 mb-4 border-b border-white/5 pb-4">
                <div className="w-3 h-3 rounded-full bg-red-500"></div>
                <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
                <div className="w-3 h-3 rounded-full bg-green-500"></div>
                <span className="ml-2 text-xs font-mono text-slate-500">bash</span>
              </div>
              <div className="font-mono text-sm leading-relaxed overflow-x-auto">
                <p className="text-cyan-400"><span className="text-pink-500">~</span>$ curl -fsSL https://get.neuralops.io | bash</p>
                <p className="text-slate-400 mt-2">Downloading NeuralOps Agent v2.4.0...</p>
                <p className="text-slate-400">Extracting binaries...</p>
                <p className="text-emerald-400 mt-2">✔ Successfully installed.</p>
                <p className="text-cyan-400 mt-4"><span className="text-pink-500">~</span>$ neuralops start --gpu all</p>
                <p className="text-slate-400 mt-2">[INFO] Initializing NVML...</p>
                <p className="text-emerald-400">[OK] Found 1x NVIDIA RTX 4090</p>
                <p className="text-slate-400">[INFO] Connecting to Federation WebSocket...</p>
                <p className="text-emerald-400 font-bold mt-2">🚀 Node Online. Ready to receive SVD jobs.</p>
                <span className="inline-block w-2 h-4 bg-white animate-pulse mt-2"></span>
              </div>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="relative z-10 border-t border-white/5 bg-[#030712] py-12 text-sm text-slate-500">
        <div className="max-w-7xl mx-auto px-6 flex flex-col md:flex-row justify-between items-center gap-6">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded bg-gradient-to-br from-cyan-400 to-blue-600 flex items-center justify-center font-black text-black text-xs">Σ</div>
            <span className="font-bold text-slate-300">NeuralOps Foundation</span>
          </div>
          
          <div className="flex gap-8">
            <Link to="/docs" className="hover:text-cyan-400 transition-colors">Documentation</Link>
            <Link to="/marketplace" className="hover:text-cyan-400 transition-colors">Marketplace</Link>
            <Link to="/paper/editor" className="hover:text-cyan-400 transition-colors">Research Paper</Link>
            <a href="https://github.com/tahamajs/SVD_linear_Attention" target="_blank" rel="noreferrer" className="hover:text-cyan-400 transition-colors">GitHub</a>
          </div>
          
          <div>
            &copy; 2026 NeuralOps. All rights reserved.
          </div>
        </div>
      </footer>
    </div>
  );
}
