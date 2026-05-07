import { Button } from '../../../components/ui/Button'

interface PDFPreviewPanelProps {
  pdfUrl: string | null
  compiling: boolean
}

export default function PDFPreviewPanel({ pdfUrl, compiling }: PDFPreviewPanelProps) {
  return (
    <div className="h-full flex flex-col">
      <div className="bg-cyber-800 border-b border-white/5 px-3 py-1 text-xs text-slate-400 flex items-center justify-between">
        <span>PDF Preview</span>
        {pdfUrl && (
          <div className="flex gap-2">
            <button
              onClick={() => window.open(pdfUrl, '_blank')}
              className="text-neon-blue hover:text-white text-xs"
            >
              Open in new tab
            </button>
            <a href={pdfUrl} download className="text-neon-blue hover:text-white text-xs">
              Download
            </a>
          </div>
        )}
      </div>
      <div className="flex-1 bg-slate-200">
        {compiling ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-slate-500 flex items-center gap-2">
              <div className="w-4 h-4 border-2 border-slate-400 border-t-neon-blue rounded-full animate-spin" />
              Compiling...
            </div>
          </div>
        ) : pdfUrl ? (
          <iframe src={pdfUrl} className="w-full h-full border-0" title="PDF Preview" />
        ) : (
          <div className="flex items-center justify-center h-full text-slate-500">
            Compile to preview PDF
          </div>
        )}
      </div>
    </div>
  )
}