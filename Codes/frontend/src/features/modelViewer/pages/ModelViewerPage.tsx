import { useState } from 'react';
import { Model3DViewer } from '../components/Model3DViewer';
import { TerminalCard } from '../../../components/ui/TerminalCard';
import { Button } from '../../../components/ui/Button';
import { modelViewerAPI } from '../services/api';
import { useToast } from '../../../components/ui/Toast';

export default function ModelViewerPage({ onClose }: { onClose?: () => void }) {
  const [modelData, setModelData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const { success, error } = useToast();

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setLoading(true);
    try {
      const res = await modelViewerAPI.parseModel(file);
      setModelData(res.data);
      success('Model loaded', `${res.data.layers.length} layers, ${res.data.totalParams.toLocaleString()} parameters`);
    } catch (err) {
      error('Failed to parse model', 'Unsupported format or file corrupted');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="p-3 border-b border-white/10 flex justify-between items-center">
        <h2 className="text-sm font-mono text-neon-mint">Model Architecture Explorer</h2>
        {onClose && <button onClick={onClose} className="text-slate-400 hover:text-white">✕</button>}
      </div>
      <div className="p-4">
        <label className="glass-terminal px-4 py-2 rounded cursor-pointer inline-block hover:bg-white/5">
          {loading ? 'Uploading...' : '📁 Upload Model (.onnx, .pt, .pb)'}
          <input type="file" accept=".onnx,.pt,.pth,.pb" onChange={handleFileUpload} className="hidden" disabled={loading} />
        </label>
      </div>
      {modelData ? (
        <div className="flex-1 min-h-0">
          <Model3DViewer modelData={modelData} />
        </div>
      ) : (
        <div className="flex-1 flex items-center justify-center text-slate-500 text-sm">
          No model loaded. Upload a file to visualize.
        </div>
      )}
    </div>
  );
}