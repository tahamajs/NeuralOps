import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuthStore } from "../store/authStore";
import { Card } from "../../../components/ui/Card";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";
import { useToast } from "../../../components/ui/Toast";
import { Eye, EyeOff, LogIn } from "lucide-react";

export default function LoginPage() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const { login } = useAuthStore();
  const navigate = useNavigate();
  const { success, error } = useToast();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      await login(username, password);
      success("Login successful", "Redirecting to command center...");
      navigate("/dashboard");
    } catch (err) {
      error("Login failed", "Invalid credentials");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-surface cyber-grid relative overflow-hidden">
      {/* Background glowing orbs */}
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-neon-blue/20 rounded-full filter blur-[100px] pointer-events-none"></div>
      <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-purple-600/20 rounded-full filter blur-[100px] pointer-events-none"></div>
      
      <Card className="w-full max-w-md p-8 animate-scale-in border border-white/10 glass shadow-2xl relative z-10">
        <div className="text-center mb-8">
          <div className="w-16 h-16 bg-gradient-to-br from-neon-blue to-purple-500 rounded-2xl flex items-center justify-center mx-auto mb-4 shadow-[0_0_30px_rgba(56,189,248,0.5)]">
            <span className="text-white font-black text-xl">NO</span>
          </div>
          <h1 className="text-3xl font-display font-bold text-white tracking-tight">
            Welcome back
          </h1>
          <p className="text-slate-400 text-sm mt-2">
            Enter your credentials to access NeuralOps
          </p>
        </div>
        
        <form onSubmit={handleSubmit} className="space-y-5">
          <Input
            label="Username"
            id="username"
            type="text"
            placeholder="your_username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
            className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white placeholder-slate-500 focus:border-neon-blue/50 transition-colors"
          />
          
          <div className="space-y-1.5">
            <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">
              Password
            </label>
            <div className="relative">
              <input
                type={showPassword ? "text" : "password"}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                required
                className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white placeholder-slate-500 focus:border-neon-blue/50 transition-colors pr-10"
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-white transition-colors"
              >
                {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
          </div>

          <Button type="submit" className="w-full bg-gradient-to-r from-neon-blue to-blue-600 hover:from-blue-400 hover:to-blue-700 border-0 shadow-[0_0_20px_rgba(56,189,248,0.3)] hover:shadow-[0_0_30px_rgba(56,189,248,0.5)] transition-all" size="lg" disabled={loading}>
            <LogIn size={18} className="mr-2" />
            {loading ? "Authenticating..." : "Login to Workspace"}
          </Button>
          
          <div className="mt-6 text-center text-xs text-slate-500">
            Don't have an account?{" "}
            <Link
              to="/register"
              className="text-neon-blue hover:text-neon-blue/80 font-semibold transition-colors"
            >
              Create account
            </Link>
          </div>
        </form>
      </Card>
    </div>
  );
}
