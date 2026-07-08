import { Button } from "../../../components/ui/Button";

const presets: Record<string, any> = {
  fast: {
    epochs: 1,
    batch_size: 32,
    learning_rate: 5e-5,
    compression_rank: 32,
  },
  accurate: {
    epochs: 5,
    batch_size: 16,
    learning_rate: 2e-5,
    compression_rank: 64,
  },
  extreme_compress: {
    epochs: 3,
    batch_size: 8,
    learning_rate: 1e-5,
    compression_rank: 16,
  },
};

export const PresetSelector = ({
  onApply,
}: {
  onApply: (preset: any) => void;
}) => (
  <div className="space-y-2">
    <h4 className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">
      Presets
    </h4>
    <div className="flex gap-2 flex-wrap">
      {Object.entries(presets).map(([key, values]) => (
        <Button
          key={key}
          variant="secondary"
          size="sm"
          onClick={() => onApply(values)}
        >
          {key.replace("_", " ")}
        </Button>
      ))}
    </div>
  </div>
);
