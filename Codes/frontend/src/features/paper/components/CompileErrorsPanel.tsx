interface CompileErrorsPanelProps {
  errors: any[];
  onClose: () => void;
}

export default function CompileErrorsPanel({
  errors,
  onClose,
}: CompileErrorsPanelProps) {
  return (
    <div className="bg-red-900/30 border-b border-red-500/20 text-red-300 text-xs font-mono max-h-32 overflow-y-auto scrollbar-thin">
      <div className="flex items-center justify-between px-3 py-1 bg-red-900/50">
        <span className="font-bold text-red-400 uppercase tracking-wider text-[10px]">
          Compilation Errors ({errors.length})
        </span>
        <button
          onClick={onClose}
          className="text-red-400 hover:text-white text-lg leading-none"
        >
          ✕
        </button>
      </div>
      <div className="px-3 py-2 whitespace-pre-wrap">
        {typeof errors === "string"
          ? errors
          : errors.map((err, i) => <div key={i}>{err}</div>)}
      </div>
    </div>
  );
}
