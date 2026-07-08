import React from "react";

interface SliderProps {
  min?: number;
  max?: number;
  step?: number;
  value: number[];
  onValueChange: (val: number[]) => void;
  className?: string;
}

export function Slider({
  min = 0,
  max = 100,
  step = 1,
  value,
  onValueChange,
  className = "",
}: SliderProps) {
  return (
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value[0]}
      onChange={(e) => onValueChange([parseFloat(e.target.value)])}
      className={`w-full accent-indigo-500 bg-slate-800 h-2 rounded-lg cursor-pointer appearance-none ${className}`}
    />
  );
}
export default Slider;
