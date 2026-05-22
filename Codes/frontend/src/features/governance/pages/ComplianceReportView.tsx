// src/features/governance/pages/ComplianceReportView.tsx
import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { governanceAPI } from '../services/api'
import { ComplianceScorecard } from '../components/ComplianceScorecard'
import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Button } from '../../../components/ui/Button'
import { Select } from '../../../components/ui/Select'
import { Spinner } from '../../../components/ui/Spinner'
import { Badge } from '../../../components/ui/Badge'
import { Progress } from '../../../components/ui/Progress'
import { CheckCircle, XCircle, AlertCircle, FileText, Download, RefreshCw } from 'lucide-react'
import { useToast } from '../../../components/ui/Toast'

export default function ComplianceReportView({ onClose }: { onClose?: () => void }) {
  const [selectedFramework, setSelectedFramework] = useState('')
  const queryClient = useQueryClient()
  const { success, error } = useToast()

  const { data: frameworks, isLoading: frameworksLoading } = useQuery({
    queryKey: ['compliance-frameworks'],
    queryFn: () => governanceAPI.getComplianceFrameworks(),
  })

  const { data: framework, isLoading: frameworkLoading } = useQuery({
    queryKey: ['compliance-framework', selectedFramework],
    queryFn: () => governanceAPI.getComplianceFramework(selectedFramework),
    enabled: !!selectedFramework,
  })

  const runCheck = useMutation({
    mutationFn: () => governanceAPI.runComplianceCheck(selectedFramework),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['compliance-framework', selectedFramework] })
      success('Compliance check completed')
    },
  })

  const generateReport = useMutation({
    mutationFn: () => governanceAPI.generateComplianceReport(selectedFramework),
    onSuccess: (res) => {
      const url = URL.createObjectURL(res.data)
      const a = document.createElement('a')
      a.href = url
      a.download = `compliance_report_${selectedFramework}_${new Date().toISOString().slice(0, 10)}.pdf`
      a.click()
      URL.revokeObjectURL(url)
      success('Report downloaded')
    },
  })

  if (frameworksLoading) return <Spinner />

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div><h1 className="text-xl font-bold text-white flex items-center gap-2"><FileText size={20} className="text-neon-mint" /> Compliance Report View</h1><p className="text-xs text-slate-400">Track compliance against industry standards</p></div>
        <div className="flex gap-2"><Select options={frameworks?.map(f => ({ value: f.id, label: `${f.name} v${f.version}` })) || []} value={selectedFramework} onChange={e => setSelectedFramework(e.target.value)} placeholder="Select framework" className="w-48" />{onClose && <Button size="sm" variant="ghost" onClick={onClose}>✕</Button>}</div>
      </div>
      <div className="flex-1 overflow-auto p-4 space-y-6">
        {selectedFramework && framework ? (<>
          <div className="flex justify-between items-center"><ComplianceScorecard framework={framework} /><div className="flex gap-2"><Button size="sm" variant="ghost" onClick={() => runCheck.mutate()}><RefreshCw size={14} className="mr-1" /> Run Check</Button><Button size="sm" variant="ghost" onClick={() => generateReport.mutate()}><Download size={14} className="mr-1" /> Export Report</Button></div></div>
          <TerminalCard className="p-4"><h3 className="text-sm font-bold text-white mb-3">Controls Summary</h3><div className="space-y-4">{framework.controls.map(control => (<div key={control.id} className="border-b border-white/5 pb-3"><div className="flex justify-between items-start"><div><div className="flex items-center gap-2"><span className="font-mono text-xs text-slate-400">{control.id}</span><h4 className="font-medium text-white">{control.title}</h4>{control.status === 'compliant' ? <CheckCircle size={14} className="text-green-400" /> : control.status === 'non_compliant' ? <XCircle size={14} className="text-red-400" /> : <AlertCircle size={14} className="text-yellow-400" />}</div><p className="text-sm text-slate-400 mt-1">{control.description}</p><div className="flex flex-wrap gap-2 mt-2"><Badge variant={control.status === 'compliant' ? 'success' : control.status === 'partial' ? 'warning' : 'danger'}>{control.status}</Badge><Badge variant="info">Risk: {control.risk_level}</Badge></div></div><div className="text-right text-xs text-slate-500"><div>Last checked: {new Date(control.last_checked).toLocaleDateString()}</div><div>Evidence: {control.evidence.length} items</div></div></div>{control.remediation_steps && <div className="mt-2 p-2 bg-yellow-500/10 rounded text-xs text-yellow-400"><strong>Remediation:</strong> {control.remediation_steps}</div>}<Progress value={control.status === 'compliant' ? 100 : control.status === 'partial' ? 50 : 0} className="mt-2 h-1" /></div>))}</div></TerminalCard>
        </>) : <div className="text-center text-slate-500 py-12">Select a compliance framework to view report</div>}
      </div>
    </div>
  )
}