import React from "react";

interface TooltipProps {
  content: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}

export function Tooltip({ content, children, className = "" }: TooltipProps) {
  return (
    <div className={`relative group inline-block ${className}`}>
      {children}
      <div className="absolute z-50 hidden group-hover:block bg-slate-900 text-xs text-slate-200 rounded-lg p-2 border border-slate-800 shadow-xl -top-10 left-1/2 -translate-x-1/2 whitespace-nowrap">
        {content}
      </div>
    </div>
  );
}
export default Tooltip;
