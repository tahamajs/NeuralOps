// src/features/researchDiscovery/pages/LiteratureSynthesizer.tsx
import { useState, useEffect } from 'react'
import { useMutation } from '@tanstack/react-query'
import { researchDiscoveryAPI } from '../services/api'
import { PaperCard } from '../components/PaperCard'
import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Button } from '../../../components/ui/Button'
import { Textarea } from '../../../components/ui/Input'
import { Spinner } from '../../../components/ui/Spinner'
import { Tabs, TabList, Tab, TabPanel } from '../../../components/ui/Tabs'
import { Brain, FileText, TrendingUp, Lightbulb, Sparkles } from 'lucide-react'
import { useToast } from '../../../components/ui/Toast'

export default function LiteratureSynthesizer({ onClose }: { onClose?: () => void }) {
  const [query, setQuery] = useState('')
  const [activeTab, setActiveTab] = useState('summary')
  const { success, error } = useToast()

  const synthesizeMutation = useMutation({
    mutationFn: (q: string) => researchDiscoveryAPI.synthesizeLiterature(q, 20),
    onError: () => error('Synthesis failed', 'Please try again'),
  })

  const handleSynthesize = () => {
    if (!query.trim()) return
    synthesizeMutation.mutate(query)
    success('Analysis started', 'AI is processing the literature')
  }

  const result = synthesizeMutation.data

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div><h1 className="text-xl font-bold text-white flex items-center gap-2"><Brain size={20} className="text-neon-mint" /> Literature Synthesizer</h1><p className="text-xs text-slate-400">AI‑powered literature review and gap analysis</p></div>
        {onClose && <Button size="sm" variant="ghost" onClick={onClose}>✕</Button>}
      </div>
      <div className="flex-1 overflow-auto p-4 space-y-4">
        <TerminalCard className="p-4">
          <Textarea rows={4} value={query} onChange={e => setQuery(e.target.value)} placeholder="Enter your research question or topic..." className="mb-3" />
          <Button onClick={handleSynthesize} disabled={!query.trim() || synthesizeMutation.isPending}>{synthesizeMutation.isPending ? 'Analyzing...' : 'Synthesize Literature'}</Button>
        </TerminalCard>
        {synthesizeMutation.isPending && <div className="text-center py-12"><Spinner size="lg" /><p className="text-slate-400 mt-2">AI is searching and synthesizing papers...</p></div>}
        {result && (<Tabs value={activeTab} onValueChange={setActiveTab}><TabList><Tab value="summary"><Sparkles size={14} className="mr-1" /> Summary</Tab><Tab value="findings"><Lightbulb size={14} className="mr-1" /> Key Findings</Tab><Tab value="gaps"><TrendingUp size={14} className="mr-1" /> Research Gaps</Tab><Tab value="papers"><FileText size={14} className="mr-1" /> Top Papers</Tab></TabList>
          <TabPanel value="summary"><TerminalCard className="p-4"><div className="prose prose-invert max-w-none whitespace-pre-wrap">{result.summary}</div></TerminalCard></TabPanel>
          <TabPanel value="findings"><TerminalCard className="p-4"><ul className="list-disc list-inside space-y-2">{result.key_findings.map((f, i) => <li key={i} className="text-slate-300">{f}</li>)}</ul></TerminalCard></TabPanel>
          <TabPanel value="gaps"><TerminalCard className="p-4"><ul className="list-disc list-inside space-y-2">{result.research_gaps.map((g, i) => <li key={i} className="text-yellow-400">{g}</li>)}</ul></TerminalCard></TabPanel>
          <TabPanel value="papers"><div className="space-y-3">{result.top_papers.map(paper => <PaperCard key={paper.id} paper={paper} />)}</div></TabPanel>
        </Tabs>)}
      </div>
    </div>
  )
}