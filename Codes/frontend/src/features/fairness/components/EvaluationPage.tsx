import React, { useState } from "react";
import { Card } from "../../../components/ui/Card";
import { Button } from "../../../components/ui/Button";
import { Shield, Activity, BarChart, AlertTriangle, CheckCircle } from "lucide-react";

export const EvaluationPage = ({ type }: { type: string }) => {
  const [isRunning, setIsRunning] = useState(false);
  const [results, setResults] = useState<any>(null);

  const runEvaluation = () => {
    setIsRunning(true);
    setTimeout(() => {
      setResults({
        score: 92,
        issues: 2,
        details: [
          { metric: "Demographic Parity", value: "0.95", status: "pass" },
          { metric: "Equal Opportunity", value: "0.88", status: "warn" },
          { metric: "Disparate Impact", value: "0.91", status: "pass" },
        ]
      });
      setIsRunning(false);
    }, 2000);
  };

  return (
    <div className="space-y-6 animate-fade-in pb-10">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-display font-bold text-white capitalize">{type} Evaluation</h1>
          <p className="text-slate-400 mt-1">Analyze model performance for potential biases and robustness issues.</p>
        </div>
        <Button 
          onClick={runEvaluation} 
          disabled={isRunning}
          className="bg-neon-blue text-black hover:bg-neon-blue/80"
        >
          {isRunning ? "Evaluating..." : "Run Evaluation"}
        </Button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card className="glass p-6 border-white/5 flex flex-col items-center justify-center text-center">
          <div className="w-16 h-16 rounded-full bg-blue-500/10 flex items-center justify-center mb-4">
            <Shield className="text-neon-blue w-8 h-8" />
          </div>
          <h3 className="text-lg font-semibold text-white">Overall Score</h3>
          <p className="text-4xl font-display font-bold text-neon-blue mt-2">
            {results ? `${results.score}%` : "--"}
          </p>
        </Card>

        <Card className="glass p-6 border-white/5 flex flex-col items-center justify-center text-center">
          <div className="w-16 h-16 rounded-full bg-amber-500/10 flex items-center justify-center mb-4">
            <AlertTriangle className="text-amber-400 w-8 h-8" />
          </div>
          <h3 className="text-lg font-semibold text-white">Detected Issues</h3>
          <p className="text-4xl font-display font-bold text-amber-400 mt-2">
            {results ? results.issues : "--"}
          </p>
        </Card>

        <Card className="glass p-6 border-white/5 flex flex-col items-center justify-center text-center">
          <div className="w-16 h-16 rounded-full bg-green-500/10 flex items-center justify-center mb-4">
            <Activity className="text-green-400 w-8 h-8" />
          </div>
          <h3 className="text-lg font-semibold text-white">Status</h3>
          <p className="text-2xl font-semibold text-green-400 mt-4">
            {results ? "Review Required" : "Ready"}
          </p>
        </Card>
      </div>

      {results && (
        <Card className="glass border-white/5 p-6">
          <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
            <BarChart className="text-neon-blue" /> Detailed Metrics
          </h2>
          <div className="space-y-4">
            {results.details.map((detail: any, idx: number) => (
              <div key={idx} className="flex items-center justify-between p-4 rounded-xl bg-white/5 border border-white/5">
                <div>
                  <h4 className="text-white font-medium">{detail.metric}</h4>
                  <p className="text-sm text-slate-400">Score: {detail.value}</p>
                </div>
                <div>
                  {detail.status === 'pass' ? (
                    <span className="flex items-center gap-1 text-green-400 text-sm font-medium bg-green-400/10 px-3 py-1 rounded-full">
                      <CheckCircle size={14} /> Passed
                    </span>
                  ) : (
                    <span className="flex items-center gap-1 text-amber-400 text-sm font-medium bg-amber-400/10 px-3 py-1 rounded-full">
                      <AlertTriangle size={14} /> Warning
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
};
