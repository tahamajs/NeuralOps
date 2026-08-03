import React from "react";

interface DateRangePickerProps {
  value?: { from?: Date; to?: Date };
  onChange?: (val: { from?: Date; to?: Date }) => void;
  className?: string;
}

export function DateRangePicker({
  value,
  onChange,
  className = "",
}: DateRangePickerProps) {
  return (
    <div className={`flex items-center gap-2 ${className}`}>
      <input
        type="date"
        className="bg-slate-800 text-slate-200 border border-slate-700 rounded p-2"
        value={value?.from?.toISOString().split("T")[0] || ""}
        onChange={(e) =>
          onChange?.({ from: new Date(e.target.value), to: value?.to })
        }
      />
      <span className="text-slate-400">to</span>
      <input
        type="date"
        className="bg-slate-800 text-slate-200 border border-slate-700 rounded p-2"
        value={value?.to?.toISOString().split("T")[0] || ""}
        onChange={(e) =>
          onChange?.({ from: value?.from, to: new Date(e.target.value) })
        }
      />
    </div>
  );
}
export default DateRangePicker;
