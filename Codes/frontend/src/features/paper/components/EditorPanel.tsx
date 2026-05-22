import { useRef } from 'react'
import MonacoEditor from '@monaco-editor/react'

interface EditorPanelProps {
  content: string
  onChange: (value: string) => void
  language: string
  onSave: () => void
}

export function EditorPanel({ content, onChange, language, onSave }: EditorPanelProps) {
  const editorRef = useRef<any>()

  const handleMount = (editor: any) => {
    editorRef.current = editor
    editor.addCommand(259 /* Ctrl+S */, onSave) // Save on Ctrl+S
  }

  return (
    <MonacoEditor
      height="100%"
      language={language}
      theme="vs-dark"
      value={content}
      onChange={(value) => onChange(value || '')}
      onMount={handleMount}
      options={{
        fontSize: 13,
        fontFamily: "'JetBrains Mono', monospace",
        minimap: { enabled: false },
        wordWrap: 'on',
        lineNumbers: 'on',
        automaticLayout: true,
        scrollBeyondLastLine: false,
        bracketPairColorization: { enabled: true },
        renderWhitespace: 'selection',
        quickSuggestions: true,
        snippetSuggestions: 'inline',
      }}
    />
  )
}