import { useNavigate } from "react-router-dom";
import { Button } from "../components/ui/Button";
import { ArrowLeft, Home, Search, AlertTriangle } from "lucide-react";

export default function NotFoundPage() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen flex items-center justify-center bg-surface cyber-grid relative overflow-hidden p-6">
      {/* Background glowing orbs */}
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-red-500/10 rounded-full filter blur-[100px] pointer-events-none"></div>
      <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-purple-600/10 rounded-full filter blur-[100px] pointer-events-none"></div>

      <div className="max-w-2xl w-full text-center relative z-10 animate-fade-in">
        <div className="w-24 h-24 bg-red-500/10 border border-red-500/20 rounded-3xl flex items-center justify-center mx-auto mb-8 shadow-[0_0_30px_rgba(239,68,68,0.2)]">
          <AlertTriangle size={40} className="text-red-400" />
        </div>

        <h1 className="text-8xl font-display font-black text-transparent bg-clip-text bg-gradient-to-br from-white to-slate-500 tracking-tighter mb-4">
          404
        </h1>
        <h2 className="text-2xl font-bold text-white mb-4">
          Anomaly Detected: Page Not Found
        </h2>
        <p className="text-slate-400 text-lg mb-10 max-w-lg mx-auto">
          The research endpoint or platform resource you requested has either
          been moved, deleted, or never existed in the NeuralOps continuum.
        </p>

        <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
          <Button
            variant="ghost"
            onClick={() => navigate(-1)}
            className="w-full sm:w-auto flex items-center justify-center gap-2"
          >
            <ArrowLeft size={18} /> Go Back
          </Button>
          <Button
            onClick={() => navigate("/dashboard")}
            className="w-full sm:w-auto flex items-center justify-center gap-2 bg-gradient-to-r from-neon-blue to-purple-500 border-0 shadow-[0_0_20px_rgba(56,189,248,0.3)] hover:shadow-[0_0_30px_rgba(56,189,248,0.5)]"
          >
            <Home size={18} /> Command Center
          </Button>
        </div>

        {/* Quick Links Section */}
        <div className="mt-16 text-left border-t border-white/10 pt-10">
          <p className="text-sm text-slate-500 uppercase tracking-widest font-bold mb-4 text-center">
            Suggested Routing
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <button
              onClick={() => navigate("/launcher")}
              className="glass p-4 rounded-xl text-center hover:bg-white/10 transition-colors"
            >
              <p className="text-white font-medium text-sm">Launch Pad</p>
              <p className="text-xs text-slate-400 mt-1">Start a new run</p>
            </button>
            <button
              onClick={() => navigate("/experiments")}
              className="glass p-4 rounded-xl text-center hover:bg-white/10 transition-colors"
            >
              <p className="text-white font-medium text-sm">Experiments</p>
              <p className="text-xs text-slate-400 mt-1">View active traces</p>
            </button>
            <button
              onClick={() => navigate("/marketplace")}
              className="glass p-4 rounded-xl text-center hover:bg-white/10 transition-colors"
            >
              <p className="text-white font-medium text-sm">GPU Hub</p>
              <p className="text-xs text-slate-400 mt-1">Acquire compute</p>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
