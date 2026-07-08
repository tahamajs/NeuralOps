import React from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";

interface PaginationControlsProps {
  currentPage: number;
  totalPages: number;
  onPageChange: (page: number) => void;
  className?: string;
}

export function PaginationControls({
  currentPage,
  totalPages,
  onPageChange,
  className = "",
}: PaginationControlsProps) {
  return (
    <div className={`flex items-center gap-2 ${className}`}>
      <button
        onClick={() => onPageChange(currentPage - 1)}
        disabled={currentPage <= 1}
        className="p-2 rounded bg-slate-800 text-slate-200 hover:bg-slate-700 disabled:opacity-50"
      >
        <ChevronLeft size={16} />
      </button>
      <span className="text-sm text-slate-400">
        Page {currentPage} of {totalPages}
      </span>
      <button
        onClick={() => onPageChange(currentPage + 1)}
        disabled={currentPage >= totalPages}
        className="p-2 rounded bg-slate-800 text-slate-200 hover:bg-slate-700 disabled:opacity-50"
      >
        <ChevronRight size={16} />
      </button>
    </div>
  );
}
export default PaginationControls;
