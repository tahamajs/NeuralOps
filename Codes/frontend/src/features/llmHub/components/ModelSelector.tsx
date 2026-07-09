import React from "react";
import { LLMProviderConfig } from "../types";

interface ModelSelectorProps {
  providers: LLMProviderConfig[];
  selectedProvider: string;
  setSelectedProvider: (provider: string) => void;
  selectedModel: string;
  setSelectedModel: (model: string) => void;
  models: string[];
}

export function ModelSelector({
  providers,
  selectedProvider,
  setSelectedProvider,
  selectedModel,
  setSelectedModel,
  models,
}: ModelSelectorProps) {
  return (
    <div className="flex flex-col sm:flex-row gap-3 w-full">
      <div className="flex-1">
        <label className="block text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1">
          Provider
        </label>
        <select
          value={selectedProvider}
          onChange={(e) => {
            const nextProv = e.target.value;
            setSelectedProvider(nextProv);
            const found = providers.find((p) => p.type === nextProv);
            if (found && found.models.length > 0) {
              setSelectedModel(found.models[0]);
            }
          }}
          className="w-full bg-slate-900 border border-white/10 rounded-xl px-3 py-2 text-xs text-white focus:border-neon-blue/50 focus:outline-none transition-colors"
        >
          {providers.map((p) => (
            <option key={p.id} value={p.type} className="bg-slate-950">
              {p.name}
            </option>
          ))}
        </select>
      </div>

      <div className="flex-1">
        <label className="block text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1">
          Model
        </label>
        <select
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
          className="w-full bg-slate-900 border border-white/10 rounded-xl px-3 py-2 text-xs text-white focus:border-neon-blue/50 focus:outline-none transition-colors"
        >
          {models.map((model) => (
            <option key={model} value={model} className="bg-slate-950">
              {model}
            </option>
          ))}
          {models.length === 0 && (
            <option value="" disabled className="bg-slate-950">
              No models available
            </option>
          )}
        </select>
      </div>
    </div>
  );
}
