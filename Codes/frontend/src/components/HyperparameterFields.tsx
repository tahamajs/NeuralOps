export default function HyperparameterFields({
  values,
  onChange,
}: {
  values: any
  onChange: (key: string, value: any) => void
}) {
  return (
    <>
      <div>
        <label className="block text-sm font-medium">Epochs</label>
        <input type="number" className="w-full border rounded px-3 py-2" value={values.epochs} onChange={(e) => onChange('epochs', parseInt(e.target.value))} />
      </div>
      <div>
        <label className="block text-sm font-medium">Batch Size</label>
        <input type="number" className="w-full border rounded px-3 py-2" value={values.batch_size} onChange={(e) => onChange('batch_size', parseInt(e.target.value))} />
      </div>
      <div>
        <label className="block text-sm font-medium">Learning Rate</label>
        <input type="number" step="any" className="w-full border rounded px-3 py-2" value={values.learning_rate} onChange={(e) => onChange('learning_rate', parseFloat(e.target.value))} />
      </div>
      <div>
        <label className="block text-sm font-medium">Compression Rank</label>
        <input type="number" className="w-full border rounded px-3 py-2" value={values.compression_rank} onChange={(e) => onChange('compression_rank', parseInt(e.target.value))} />
      </div>
      <div>
        <label className="block text-sm font-medium">Max Sequence Length</label>
        <input type="number" className="w-full border rounded px-3 py-2" value={values.max_seq_length} onChange={(e) => onChange('max_seq_length', parseInt(e.target.value))} />
      </div>
    </>
  )
}