import { useState } from 'react'
import { FileText, Loader2 } from 'lucide-react'

export const DownloadPaperButton = ({ experimentId }: { experimentId: string }) => {
  const [loading, setLoading] = useState(false)

  const handleDownload = async () => {
    setLoading(true)
    try {
      const response = await fetch(`/api/runs/${experimentId}/paper/`)
      if (!response.ok) throw new Error('Failed')
      const blob = await response.blob()
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      const ext = response.headers.get('content-type')?.includes('pdf') ? 'pdf' : 'tex'
      a.download = `paper_${experimentId}.${ext}`
      a.click()
      URL.revokeObjectURL(url)
    } catch {
      alert('Paper generation failed.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-2">
      <p className="text-xs text-slate-400">
        Generate a LaTeX paper from this experiment's results. Compiles to PDF.
      </p>
      <button
        onClick={handleDownload}
        disabled={loading}
        className="inline-flex items-center gap-2 bg-purple-500/10 border border-purple-500/30 text-purple-400 px-4 py-2 rounded-xl text-sm font-semibold hover:bg-purple-500/20 transition-colors disabled:opacity-50"
      >
        {loading ? <Loader2 size={16} className="animate-spin" /> : <FileText size={16} />}
        {loading ? 'Generating...' : '📄 Generate Paper'}
      </button>
    </div>
  )
}