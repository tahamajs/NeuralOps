import { useState, useEffect } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { paperAPI } from '../services/api'
import { Spinner, Button, Tabs, TabList, Tab, TabPanel } from '../../../components/ui'
import { PdfPreview } from './PdfPreview'
import { Play, RefreshCw } from 'lucide-react'

interface CompilationPanelProps {
  projectId: string
}

export function CompilationPanel({ projectId }: CompilationPanelProps) {
  const [pdfUrl, setPdfUrl] = useState<string | null>(null)
  const [compilationLogs, setCompilationLogs] = useState('')
  const [activeTab, setActiveTab] = useState('preview')

  const compile = useMutation({
    mutationFn: () => paperAPI.compile(projectId),
    onSuccess: async (data) => {
      // Poll for status or wait for completion
      await new Promise(resolve => setTimeout(resolve, 2000))
      const url = await paperAPI.getPdf(projectId)
      setPdfUrl(url)
      const logs = await paperAPI.getCompilationLogs(projectId)
      setCompilationLogs(logs)
    },
  })

  const { data: logsData } = useQuery({
    queryKey: ['paper-logs', projectId],
    queryFn: () => paperAPI.getCompilationLogs(projectId),
    enabled: activeTab === 'logs',
  })

  return (
    <div className="flex flex-col h-full">
      <div className="p-3 border-b border-white/10 flex justify-between items-center">
        <h3 className="text-sm font-bold text-white">Output</h3>
        <div className="flex gap-2">
          <Button size="sm" variant="ghost" onClick={() => compile.mutate()} disabled={compile.isPending}>
            {compile.isPending ? <RefreshCw size={14} className="animate-spin" /> : <Play size={14} />} Compile
          </Button>
        </div>
      </div>
      <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 flex flex-col">
        <TabList>
          <Tab value="preview">Preview</Tab>
          <Tab value="logs">Logs</Tab>
        </TabList>
        <TabPanel value="preview" className="flex-1 p-2">
          <PdfPreview pdfUrl={pdfUrl} isLoading={compile.isPending} error={null} />
        </TabPanel>
        <TabPanel value="logs" className="flex-1 p-2">
          <pre className="text-xs text-slate-300 whitespace-pre-wrap font-mono">{logsData || 'No compilation logs yet'}</pre>
        </TabPanel>
      </Tabs>
    </div>
  )
}