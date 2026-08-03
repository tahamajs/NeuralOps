import React from "react";
import { Sliders, HelpCircle } from "lucide-react";

interface ParameterControlsProps {
  parameters: {
    temperature: number;
    max_tokens: number;
    top_p: number;
  };
  onChange: (params: any) => void;
}

export function ParameterControls({
  parameters,
  onChange,
}: ParameterControlsProps) {
  const updateParam = (key: string, value: number) => {
    onChange({
      ...parameters,
      [key]: value,
    });
  };

  return (
    <div className="flex flex-col gap-3 p-3.5 bg-slate-900/60 border border-white/5 rounded-xl text-xs w-full sm:w-80 max-w-sm shrink-0">
      <div className="flex items-center gap-1.5 border-b border-white/5 pb-2 text-white font-bold mb-1">
        <Sliders size={14} className="text-neon-blue" />
        Parameters Configuration
      </div>

      <div className="space-y-3">
        {/* Temperature */}
        <div>
          <div className="flex justify-between items-center mb-1 text-slate-400">
            <span className="flex items-center gap-1">
              Temperature
              <span title="Controls randomness of output">
                <HelpCircle size={10} className="text-slate-600" />
              </span>
            </span>
            <span className="font-mono text-neon-blue font-bold">
              {parameters.temperature.toFixed(2)}
            </span>
          </div>
          <input
            type="range"
            min="0"
            max="1.5"
            step="0.05"
            value={parameters.temperature}
            onChange={(e) =>
              updateParam("temperature", parseFloat(e.target.value))
            }
            className="w-full accent-neon-blue h-1 bg-slate-950 rounded-lg appearance-none cursor-pointer"
          />
        </div>

        {/* Max Tokens */}
        <div>
          <div className="flex justify-between items-center mb-1 text-slate-400">
            <span className="flex items-center gap-1">
              Max Tokens
              <span title="Max number of tokens to generate">
                <HelpCircle size={10} className="text-slate-600" />
              </span>
            </span>
            <span className="font-mono text-purple-400 font-bold">
              {parameters.max_tokens}
            </span>
          </div>
          <input
            type="range"
            min="10"
            max="4000"
            step="50"
            value={parameters.max_tokens}
            onChange={(e) =>
              updateParam("max_tokens", parseInt(e.target.value))
            }
            className="w-full accent-purple-500 h-1 bg-slate-950 rounded-lg appearance-none cursor-pointer"
          />
        </div>

        {/* Top P */}
        <div>
          <div className="flex justify-between items-center mb-1 text-slate-400">
            <span className="flex items-center gap-1">
              Top P
              <span title="Nucleus sampling threshold">
                <HelpCircle size={10} className="text-slate-600" />
              </span>
            </span>
            <span className="font-mono text-emerald-400 font-bold">
              {parameters.top_p.toFixed(2)}
            </span>
          </div>
          <input
            type="range"
            min="0.05"
            max="1.0"
            step="0.05"
            value={parameters.top_p}
            onChange={(e) => updateParam("top_p", parseFloat(e.target.value))}
            className="w-full accent-emerald-500 h-1 bg-slate-950 rounded-lg appearance-none cursor-pointer"
          />
        </div>
      </div>
    </div>
  );
}
