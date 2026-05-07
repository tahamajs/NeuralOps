import { useRef, useEffect } from 'react'
import Editor, { OnMount } from '@monaco-editor/react'

interface MonacoEditorProps {
  value: string
  onChange: (value: string | undefined) => void
  editorRef: React.MutableRefObject<any>
}

export default function MonacoEditorWrapper({ value, onChange, editorRef }: MonacoEditorProps) {
  const handleMount: OnMount = (editor) => {
    editorRef.current = editor
  }

  return (
    <Editor
      height="100%"
      defaultLanguage="latex"
      theme="vs-dark"
      value={value}
      onChange={onChange}
      onMount={handleMount}
      options={{
        fontSize: 14,
        fontFamily: "'JetBrains Mono', monospace",
        minimap: { enabled: false },
        wordWrap: 'on',
        scrollBeyondLastLine: false,
        automaticLayout: true,
        tabSize: 2,
        padding: { top: 8 },
      }}
    />
  )
}