import { Card } from "../../../components/ui/Card";
import { Button } from "../../../components/ui/Button";
import {
  Shield,
  ShieldAlert,
  Target,
  Activity,
  CheckCircle,
  Search,
} from "lucide-react";

export default function RobustnessPage() {
  const attacks = [
    {
      name: "PGD (Projected Gradient Descent)",
      severity: "High",
      successRate: "87%",
      status: "vulnerable",
    },
    {
      name: "FGSM (Fast Gradient Sign)",
      severity: "Medium",
      successRate: "42%",
      status: "vulnerable",
    },
    {
      name: "Carlini-Wagner L2",
      severity: "Critical",
      successRate: "94%",
      status: "vulnerable",
    },
    {
      name: "DeepFool",
      severity: "High",
      successRate: "12%",
      status: "secure",
    },
  ];

  return (
    <div className="p-6 space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-2xl font-display font-bold text-white flex items-center gap-2">
            <Shield className="text-rose-500" size={24} /> Adversarial
            Robustness
          </h1>
          <p className="text-slate-400 mt-1">
            Simulate adversarial attacks and measure model resilience
          </p>
        </div>
        <Button className="flex items-center gap-2 bg-rose-500/20 text-rose-400 border border-rose-500/50 hover:bg-rose-500/30">
          <Target size={16} /> Run Attack Simulation
        </Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Stats */}
        <div className="space-y-6">
          <Card className="glass p-5 border-rose-500/20 shadow-[0_0_15px_rgba(244,63,94,0.1)]">
            <h3 className="text-sm uppercase tracking-wider font-bold text-slate-400 mb-4">
              Resilience Score
            </h3>
            <div className="flex items-end gap-3 mb-2">
              <span className="text-5xl font-display font-black text-white">
                42<span className="text-2xl text-slate-500">/100</span>
              </span>
            </div>
            <p className="text-xs text-rose-400 flex items-center gap-1">
              <ShieldAlert size={14} /> High vulnerability detected
            </p>
            <div className="mt-4 h-2 w-full bg-white/10 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-red-500 to-rose-400"
                style={{ width: "42%" }}
              ></div>
            </div>
          </Card>

          <Card className="glass p-5">
            <h3 className="text-sm uppercase tracking-wider font-bold text-slate-400 mb-4">
              Attack History
            </h3>
            <div className="space-y-4">
              <div className="flex justify-between items-center text-sm">
                <span className="text-slate-300">Total Simulations</span>
                <span className="text-white font-mono">1,024</span>
              </div>
              <div className="flex justify-between items-center text-sm">
                <span className="text-slate-300">Successful Attacks</span>
                <span className="text-rose-400 font-mono">680</span>
              </div>
              <div className="flex justify-between items-center text-sm">
                <span className="text-slate-300">Defeated Attacks</span>
                <span className="text-green-400 font-mono">344</span>
              </div>
            </div>
          </Card>
        </div>

        {/* Right Column: Attack Vectors */}
        <div className="lg:col-span-2">
          <Card className="glass overflow-hidden">
            <div className="p-4 border-b border-white/5 flex items-center justify-between">
              <h3 className="font-semibold text-white flex items-center gap-2">
                <Activity size={18} className="text-neon-blue" /> Attack Vectors
              </h3>
              <div className="relative">
                <Search
                  size={14}
                  className="absolute left-2.5 top-1/2 -translate-y-1/2 text-slate-400"
                />
                <input
                  type="text"
                  placeholder="Filter attacks..."
                  className="pl-8 pr-3 py-1 bg-white/5 border border-white/10 rounded text-xs text-white focus:outline-none focus:border-neon-blue/50"
                />
              </div>
            </div>
            <div className="divide-y divide-white/5">
              {attacks.map((atk, i) => (
                <div
                  key={i}
                  className="p-4 flex items-center justify-between hover:bg-white/5 transition-colors"
                >
                  <div className="flex items-center gap-3">
                    {atk.status === "secure" ? (
                      <CheckCircle size={18} className="text-green-400" />
                    ) : (
                      <ShieldAlert size={18} className="text-rose-400" />
                    )}
                    <div>
                      <p className="text-sm font-medium text-white">
                        {atk.name}
                      </p>
                      <p className="text-xs text-slate-500">
                        Severity: {atk.severity}
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-xs text-slate-400 uppercase">
                      Success Rate
                    </p>
                    <p
                      className={`text-sm font-mono font-bold ${atk.status === "secure" ? "text-green-400" : "text-rose-400"}`}
                    >
                      {atk.successRate}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
