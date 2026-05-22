import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { paperAPI } from '../services/api'
import { Modal, Button } from '../../../components/ui'
import { Download, Archive, FileJson } from 'lucide-react'

interface ExportModalProps {
  open: boolean
  onClose: () => void
  projectId: string
}

export function ExportModal({ open, onClose, projectId }: ExportModalProps) {
  const [exportFormat, setExportFormat] = useState('zip')

  const exportProject = useMutation({
    mutationFn: () => paperAPI.exportProject(projectId),
    onSuccess: (url) => {
      const a = document.createElement('a')
      a.href = url
      a.download = `project-${projectId}.zip`
      a.click()
      URL.revokeObjectURL(url)
      onClose()
    },
  })

  return (
    <Modal open={open} onClose={onClose} title="Export Project">
      <div className="space-y-4">
        <div className="flex gap-3">
          <button onClick={() => setExportFormat('zip')} className={`flex-1 p-4 rounded-xl border ${exportFormat === 'zip' ? 'border-neon-blue bg-neon-blue/10' : 'border-white/10'}`}>
            <Archive size={24} className="mx-auto mb-2 text-neon-blue" />
            <div className="text-sm font-medium text-white">ZIP Archive</div>
            <div className="text-xs text-slate-400">Complete project folder</div>
          </button>
          <button onClick={() => setExportFormat('overleaf')} className={`flex-1 p-4 rounded-xl border ${exportFormat === 'overleaf' ? 'border-neon-blue bg-neon-blue/10' : 'border-white/10'}`}>
            <FileJson size={24} className="mx-auto mb-2 text-neon-blue" />
            <div className="text-sm font-medium text-white">Overleaf</div>
            <div className="text-xs text-slate-400">Overleaf-compatible ZIP</div>
          </button>
        </div>
        <Button onClick={() => exportProject.mutate()} disabled={exportProject.isPending} className="w-full">
          <Download size={14} className="mr-1" /> Export {exportFormat === 'zip' ? 'ZIP Archive' : 'Overleaf Archive'}
        </Button>
      </div>
    </Modal>
  )
}