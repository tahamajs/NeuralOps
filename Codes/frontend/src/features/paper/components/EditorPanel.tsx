import Editor, { OnMount } from '@monaco-editor/react'

interface EditorPanelProps {
  value: string
  onChange: (value: string) => void
  editorRef: React.MutableRefObject<any>
  language: string
  fileName?: string
}

export default function EditorPanel({ value, onChange, editorRef, language, fileName }: EditorPanelProps) {
  const handleMount: OnMount = (editor) => {
    editorRef.current = editor
  }

  return (
    <div className="h-full flex flex-col">
      <div className="bg-cyber-800 border-b border-white/5 px-3 py-1 text-xs text-slate-400 flex items-center justify-between">
        <span>{fileName || 'untitled'}</span>
        <span className="text-[10px] text-slate-500">LaTeX</span>
      </div>
      <div className="flex-1">
        <Editor
          height="100%"
          language={language}
          theme="vs-dark"
          value={value}
          onChange={(val) => onChange(val || '')}
          onMount={handleMount}
          options={{
            fontSize: 13,
            fontFamily: "'JetBrains Mono', monospace",
            minimap: { enabled: false },
            wordWrap: 'on',
            scrollBeyondLastLine: false,
            automaticLayout: true,
            tabSize: 2,
            lineNumbers: 'on',
            renderWhitespace: 'selection',
            bracketPairColorization: { enabled: true },
            padding: { top: 16 },
          }}
        />
      </div>
    </div>
  )
}