export default function PresetSelector({ onApply }: { onApply: (preset: any) => void }) {
  const presets: Record<string, any> = {
    fast: { epochs: 1, batch_size: 32, learning_rate: 5e-5, compression_rank: 32 },
    accurate: { epochs: 5, batch_size: 16, learning_rate: 2e-5, compression_rank: 64 },
    extreme_compress: { epochs: 3, batch_size: 8, learning_rate: 1e-5, compression_rank: 16 },
  }

  return (
    <div>
      <label className="block text-sm font-medium">Preset</label>
      <select className="w-full border rounded px-3 py-2" onChange={(e) => onApply(presets[e.target.value] || {})}>
        <option value="">None</option>
        <option value="fast">Fast</option>
        <option value="accurate">Accurate</option>
        <option value="extreme_compress">Extreme Compress</option>
      </select>
    </div>
  )
}