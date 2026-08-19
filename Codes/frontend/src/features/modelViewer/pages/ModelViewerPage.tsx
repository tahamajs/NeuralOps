import { useState, useEffect } from "react";
import { Model3DViewer } from "../components/Model3DViewer";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Select } from "../../../components/ui/Select";
import { modelViewerAPI } from "../services/api";
import { useToast } from "../../../components/ui/Toast";
import { Box, Upload, Layers, Cpu } from "lucide-react";
import type { ModelArchitecture } from "../types";

const PRESET_MODELS: Record<string, ModelArchitecture> = {
  "bert-svd": {
    id: "bert-svd",
    name: "BERT-Base (Griffin SVD Linear Attention)",
    framework: "pytorch",
    totalParams: 86400000,
    layers: [
      { name: "Embedding (Vocab 30k)", type: "Embedding", params: 23800000, color: "3b82f6" },
      { name: "Layer 1-3 SVD Projections", type: "GriffinAttention", params: 15600000, color: "00ff9d" },
      { name: "Layer 4-6 SVD Projections", type: "GriffinAttention", params: 15600000, color: "00ff9d" },
      { name: "Layer 7-9 SVD Projections", type: "GriffinAttention", params: 15600000, color: "00ff9d" },
      { name: "Layer 10-12 SVD Projections", type: "GriffinAttention", params: 15600000, color: "00ff9d" },
      { name: "FeedForward GELU (4x)", type: "DenseFFN", params: 23500000, color: "a855f7" },
      { name: "Classifier & Pooler Head", type: "Linear", params: 590000, color: "f59e0b" },
    ],
  },
  "llama-3-8b": {
    id: "llama-3-8b",
    name: "LLaMA-3 8B (Grouped-Query Attention)",
    framework: "pytorch",
    totalParams: 8030000000,
    layers: [
      { name: "Rotary Embeddings (RoPE)", type: "Embedding", params: 524000000, color: "3b82f6" },
      { name: "32x GQA Attention Blocks", type: "GQA_Attention", params: 3200000000, color: "00ff9d" },
      { name: "SwiGLU MLP Blocks", type: "SwiGLU", params: 4300000000, color: "a855f7" },
      { name: "RMSNorm & LM Output Head", type: "Linear", params: 128000000, color: "f59e0b" },
    ],
  },
  "vit-base": {
    id: "vit-base",
    name: "Vision Transformer ViT-B/16",
    framework: "pytorch",
    totalParams: 86500000,
    layers: [
      { name: "Patch Projection (16x16)", type: "Conv2d", params: 590000, color: "3b82f6" },
      { name: "Class Token & Pos Embedding", type: "Embedding", params: 768000, color: "3b82f6" },
      { name: "12x Multi-Head Self-Attention", type: "SelfAttention", params: 42000000, color: "00ff9d" },
      { name: "MLP Representation Heads", type: "DenseFFN", params: 43000000, color: "a855f7" },
      { name: "ImageNet 1k Linear Head", type: "Linear", params: 769000, color: "f59e0b" },
    ],
  },
};

export default function ModelViewerPage({ onClose }: { onClose?: () => void }) {
  const [selectedPreset, setSelectedPreset] = useState("bert-svd");
  const [modelData, setModelData] = useState<any>(PRESET_MODELS["bert-svd"]);
  const [loading, setLoading] = useState(false);
  const { success, error } = useToast();

  const handlePresetChange = (presetKey: string) => {
    setSelectedPreset(presetKey);
    if (PRESET_MODELS[presetKey]) {
      setModelData(PRESET_MODELS[presetKey]);
      success("Preset Loaded", PRESET_MODELS[presetKey].name);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setLoading(true);
    try {
      const res = await modelViewerAPI.parseModel(file);
      const parsed = res?.data || res || PRESET_MODELS["bert-svd"];
      setModelData(parsed);
      setSelectedPreset("custom");
      success(
        "Model parsed",
        `${parsed.layers?.length || 0} layers, ${(parsed.totalParams || 0).toLocaleString()} parameters`,
      );
    } catch (err) {
      error("Failed to parse model", "Using fallback architecture visualization");
      setModelData(PRESET_MODELS["bert-svd"]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-slate-950 text-white font-sans">
      <div className="p-3 border-b border-white/10 flex justify-between items-center bg-slate-900/40">
        <div className="flex items-center gap-2">
          <Box size={18} className="text-neon-mint" />
          <div>
            <h2 className="text-sm font-bold text-white">
              3D Neural Network Architecture Explorer
            </h2>
            <p className="text-[11px] text-slate-400">
              Interactive 3D layer visualization & parameter graph
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-400">Preset:</span>
            <Select
              options={[
                { value: "bert-svd", label: "BERT-Base SVD Attention" },
                { value: "llama-3-8b", label: "LLaMA-3 8B GQA" },
                { value: "vit-base", label: "Vision Transformer ViT-B/16" },
              ]}
              value={selectedPreset}
              onChange={(e) => handlePresetChange(e.target.value)}
              className="w-52 text-xs"
            />
          </div>
          <label className="glass-terminal px-3 py-1.5 rounded cursor-pointer text-xs flex items-center gap-1.5 hover:bg-white/10 text-slate-300">
            <Upload size={12} />
            {loading ? "Uploading..." : "Upload (.onnx, .pt)"}
            <input
              type="file"
              accept=".onnx,.pt,.pth,.pb"
              onChange={handleFileUpload}
              className="hidden"
              disabled={loading}
            />
          </label>
          {onClose && (
            <button onClick={onClose} className="text-slate-400 hover:text-white ml-1">
              ✕
            </button>
          )}
        </div>
      </div>

      <div className="flex-1 min-h-0 relative">
        <Model3DViewer modelData={modelData} onClose={onClose} />
      </div>
    </div>
  );
}
