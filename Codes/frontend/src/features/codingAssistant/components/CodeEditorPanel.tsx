import { useEffect, useRef, useState } from 'react'
import MonacoEditor from '@monaco-editor/react'
import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Button } from '../../../components/ui/Button'
import { Save, Loader2 } from 'lucide-react'

interface CodeEditorPanelProps {
  content: string
  language: string
  onSave: (content: string) => void
  isSaving: boolean
  readOnly?: boolean
}

export function CodeEditorPanel({ content, language, onSave, isSaving, readOnly = false }: CodeEditorPanelProps) {
  const [value, setValue] = useState(content)
  const editorRef = useRef<any>()
  useEffect(() => { setValue(content) }, [content])
  useEffect(() => { const handleCtrlS = (e: KeyboardEvent) => { if ((e.ctrlKey || e.metaKey) && e.key === 's') { e.preventDefault(); onSave(value) } }; window.addEventListener('keydown', handleCtrlS); return () => window.removeEventListener('keydown', handleCtrlS) }, [value, onSave])
  return (<TerminalCard className="h-full flex flex-col overflow-hidden"><div className="flex justify-between items-center p-2 border-b border-white/10"><div className="text-xs font-mono text-slate-400">Editor</div><Button size="sm" variant="ghost" onClick={() => onSave(value)} disabled={isSaving}>{isSaving ? <Loader2 size={14} className="animate-spin" /> : <Save size={14} />} Save</Button></div><div className="flex-1"><MonacoEditor height="100%" language={language} theme="vs-dark" value={value} onChange={(val) => setValue(val || '')} options={{ fontSize: 13, minimap: { enabled: false }, wordWrap: 'on', readOnly }} /></div></TerminalCard>)
}