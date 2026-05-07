import { useState, useCallback, useRef } from 'react'
import Editor from '@monaco-editor/react'
import { Button } from '../../../components/ui/Button'

interface PaperEditorPanelProps {
  onClose?: () => void
}

export default function PaperEditorPanel({ onClose }: PaperEditorPanelProps) {
  const [latexSource, setLatexSource] = useState(`\\documentclass{article}
\\usepackage{graphicx}
\\usepackage{amsmath}
\\usepackage{hyperref}

\\title{Experiment Report: Attention Compression}
\\author{NeuralOps Research Team}
\\date{\\today}

\\begin{document}

\\maketitle

\\section{Introduction}
This paper presents results from the attention compression experiment
using the \\texttt{griffin\_ssd} attention mechanism on the MRPC task.

\\section{Model Architecture}
The base model used is \\texttt{bert-base-uncased} with the following
modifications:
\\begin{itemize}
    \\item Attention: griffin\_ssd
    \\item Method: LoRA (rank=8)
    \\item Compression: SVD (rank=64)
\\end{itemize}

\\section{Results}
\\begin{table}[h]
\\centering
\\begin{tabular}{|c|c|}
\\hline
Metric & Value \\\\
\\hline
Accuracy & 0.852 \\\\
F1 Score & 0.891 \\\\
Loss & 0.342 \\\\
\\hline
\\end{tabular}
\\caption{Evaluation results on MRPC validation set}
\\end{table}

\\section{Conclusion}
The hybrid attention-compression approach demonstrates competitive
performance while reducing computational requirements.

\\end{document}`)

  const [pdfUrl, setPdfUrl] = useState<string | null>(null)
  const [compiling, setCompiling] = useState(false)
  const [viewMode, setViewMode] = useState<'split' | 'code' | 'preview'>('split')
  const [error, setError] = useState<string | null>(null)
  const editorRef = useRef<any>(null)

  const handleEditorMount = useCallback((editor: any) => {
    editorRef.current = editor
  }, [])

  const handleCompile = async () => {
    setCompiling(true)
    setError(null)
    try {
      const token = localStorage.getItem('token')
      const response = await fetch('/api/paper/compile/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ source: latexSource }),
      })

      if (!response.ok) {
        const errData = await response.json()
        setError(errData.error || 'Compilation failed')
        return
      }

      const blob = await response.blob()
      const url = URL.createObjectURL(blob)
      setPdfUrl(url)
    } catch (err: any) {
      setError(err.message || 'Compilation failed')
    } finally {
      setCompiling(false)
    }
  }

  const handleDownload = () => {
    const blob = new Blob([latexSource], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'paper.tex'
    a.click()
    URL.revokeObjectURL(url)
  }

  // Quick insert helpers
  const insertSnippet = (snippet: string) => {
    if (editorRef.current) {
      const selection = editorRef.current.getSelection()
      editorRef.current.executeEdits('snippet', [
        { range: selection, text: snippet },
      ])
      editorRef.current.focus()
    }
  }

  return (
    <div className="flex flex-col h-full bg-cyber-900">
      {/* Toolbar */}
      <div className="flex items-center justify-between px-3 py-1.5 bg-cyber-800 border-b border-white/5">
        <div className="flex items-center gap-1">
          <Button size="sm" variant="ghost" onClick={handleCompile} disabled={compiling}>
            {compiling ? '⏳ Compiling...' : '▶ Compile'}
          </Button>
          <Button size="sm" variant="ghost" onClick={handleDownload}>
            💾 Save .tex
          </Button>
          <div className="w-px h-4 bg-white/10 mx-1" />
          <button
            onClick={() => setViewMode('split')}
            className={`px-2 py-0.5 text-[10px] rounded ${viewMode === 'split' ? 'bg-neon-blue/20 text-neon-blue' : 'text-slate-400'}`}
          >
            Split
          </button>
          <button
            onClick={() => setViewMode('code')}
            className={`px-2 py-0.5 text-[10px] rounded ${viewMode === 'code' ? 'bg-neon-blue/20 text-neon-blue' : 'text-slate-400'}`}
          >
            Code
          </button>
          <button
            onClick={() => setViewMode('preview')}
            className={`px-2 py-0.5 text-[10px] rounded ${viewMode === 'preview' ? 'bg-neon-blue/20 text-neon-blue' : 'text-slate-400'}`}
          >
            Preview
          </button>
        </div>
        {/* Quick Insert Snippets */}
        <div className="flex items-center gap-1 text-[10px]">
          {[
            { label: 'Section', code: '\\section{}' },
            { label: 'Figure', code: '\\begin{figure}[h]\n  \\centering\n  \\includegraphics[width=0.8\\textwidth]{}\n  \\caption{}\n  \\label{fig:}\n\\end{figure}' },
            { label: 'Table', code: '\\begin{table}[h]\n  \\centering\n  \\begin{tabular}{|c|c|}\n  \\hline\n  A & B \\\\ \\hline\n  \\end{tabular}\n  \\caption{}\n\\end{table}' },
            { label: 'Math', code: '\\begin{equation}\n  \n\\end{equation}' },
            { label: 'Ref', code: '\\ref{}' },
          ].map((item) => (
            <button
              key={item.label}
              onClick={() => insertSnippet(item.code)}
              className="px-1.5 py-0.5 rounded hover:bg-white/10 text-slate-400 hover:text-white"
              title={item.label}
            >
              {item.label}
            </button>
          ))}
        </div>
      </div>

      {/* Error display */}
      {error && (
        <div className="px-3 py-2 bg-red-500/10 border-b border-red-500/20 text-red-400 text-xs font-mono whitespace-pre-wrap max-h-24 overflow-auto">
          {error}
        </div>
      )}

      {/* Editor + Preview */}
      <div className="flex-1 flex overflow-hidden">
        {/* Code Editor */}
        {(viewMode === 'split' || viewMode === 'code') && (
          <div className={`${viewMode === 'split' ? 'w-1/2' : 'w-full'} h-full border-r border-white/5`}>
            <Editor
              height="100%"
              defaultLanguage="latex"
              theme="vs-dark"
              value={latexSource}
              onChange={(val) => setLatexSource(val || '')}
              onMount={handleEditorMount}
              options={{
                fontSize: 12,
                fontFamily: "'JetBrains Mono', monospace",
                minimap: { enabled: false },
                lineNumbers: 'on',
                wordWrap: 'on',
                scrollBeyondLastLine: false,
                automaticLayout: true,
                tabSize: 2,
                renderWhitespace: 'selection',
                bracketPairColorization: { enabled: true },
                padding: { top: 8 },
              }}
            />
          </div>
        )}

        {/* PDF Preview */}
        {(viewMode === 'split' || viewMode === 'preview') && (
          <div className={`${viewMode === 'split' ? 'w-1/2' : 'w-full'} h-full bg-slate-100`}>
            {pdfUrl ? (
              <iframe src={pdfUrl} className="w-full h-full border-0" title="PDF Preview" />
            ) : (
              <div className="flex flex-col items-center justify-center h-full text-slate-400">
                <span className="text-4xl mb-4">📄</span>
                <p className="text-sm">Compile to see preview</p>
                <Button size="sm" variant="secondary" className="mt-3" onClick={handleCompile}>
                  {compiling ? 'Compiling...' : 'Compile PDF'}
                </Button>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}