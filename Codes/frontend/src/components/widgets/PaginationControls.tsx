import { ChevronLeft, ChevronRight } from 'lucide-react'

interface Props {
  currentPage: number
  totalPages: number
  onPageChange: (page: number) => void
}

export function PaginationControls({ currentPage, totalPages, onPageChange }: Props) {
  return (
    <div className="flex items-center gap-2 text-sm text-slate-400">
      <button
        onClick={() => onPageChange(currentPage - 1)}
        disabled={currentPage <= 1}
        className="p-1 hover:text-white disabled:opacity-30"
      >
        <ChevronLeft size={16} />
      </button>
      <span>
        {currentPage} / {totalPages}
      </span>
      <button
        onClick={() => onPageChange(currentPage + 1)}
        disabled={currentPage >= totalPages}
        className="p-1 hover:text-white disabled:opacity-30"
      >
        <ChevronRight size={16} />
      </button>
    </div>
  )
}