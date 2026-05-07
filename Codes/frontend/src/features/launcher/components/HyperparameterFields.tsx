import { Input } from '../../../components/ui/Input'

export const HyperparameterFields = ({
  values,
  onChange,
}: {
  values: any
  onChange: (key: string, value: number) => void
}) => (
  <div className="grid grid-cols-2 gap-4">
    <Input label="Epochs" type="number" value={values.epochs} onChange={(e) => onChange('epochs', parseInt(e.target.value))} />
    <Input label="Batch Size" type="number" value={values.batch_size} onChange={(e) => onChange('batch_size', parseInt(e.target.value))} />
    <Input label="Learning Rate" type="number" step="any" value={values.learning_rate} onChange={(e) => onChange('learning_rate', parseFloat(e.target.value))} />
    <Input label="Max Seq Length" type="number" value={values.max_seq_length} onChange={(e) => onChange('max_seq_length', parseInt(e.target.value))} />
    <Input label="Compression Rank" type="number" value={values.compression_rank} onChange={(e) => onChange('compression_rank', parseInt(e.target.value))} />
  </div>
)