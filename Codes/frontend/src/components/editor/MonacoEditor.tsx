import { Suspense, useRef, useEffect } from "react";
import Editor, { OnMount, BeforeMount } from "@monaco-editor/react";
import { Loader2 } from "lucide-react";

interface MonacoEditorProps {
  value: string;
  onChange: (value: string | undefined) => void;
  editorRef: React.MutableRefObject<any>;
  language?: string;
  readOnly?: boolean;
}

const LoadingFallback = () => (
  <div className="flex items-center justify-center h-full bg-cyber-800">
    <Loader2 size={24} className="animate-spin text-slate-400" />
  </div>
);

export default function MonacoEditorWrapper({
  value,
  onChange,
  editorRef,
  language = "latex",
  readOnly = false,
}: MonacoEditorProps) {
  const handleMount: OnMount = (editor) => {
    editorRef.current = editor;
  };

  const handleBeforeMount: BeforeMount = (monaco) => {
    // Optional: register custom LaTeX snippets or autocomplete
    monaco.languages.registerCompletionItemProvider("latex", {
      provideCompletionItems: () => ({
        suggestions: [
          {
            label: "\\section",
            kind: monaco.languages.CompletionItemKind.Snippet,
            insertText: "\\section{${1:title}}",
            insertTextRules:
              monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
            documentation: "Section heading",
          },
          {
            label: "\\begin{figure}",
            kind: monaco.languages.CompletionItemKind.Snippet,
            insertText:
              "\\begin{figure}[htbp]\n  \\centering\n  \\includegraphics[width=0.8\\textwidth]{${1:file}}\n  \\caption{${2:caption}}\n  \\label{fig:${3:label}}\n\\end{figure}",
            insertTextRules:
              monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
            documentation: "Insert a figure",
          },
          {
            label: "\\begin{table}",
            kind: monaco.languages.CompletionItemKind.Snippet,
            insertText:
              "\\begin{table}[htbp]\n  \\centering\n  \\begin{tabular}{${1:cc}}\n  ${2:data}\n  \\end{tabular}\n  \\caption{${3:caption}}\n  \\label{tab:${4:label}}\n\\end{table}",
            insertTextRules:
              monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
            documentation: "Insert a table",
          },
        ],
      }),
    });
  };

  return (
    <div className="h-full w-full">
      <Suspense fallback={<LoadingFallback />}>
        <Editor
          height="100%"
          language={language}
          theme="vs-dark"
          value={value}
          onChange={onChange}
          onMount={handleMount}
          beforeMount={handleBeforeMount}
          loading={<LoadingFallback />}
          options={{
            fontSize: 14,
            fontFamily: "'JetBrains Mono', 'Cascadia Code', monospace",
            minimap: { enabled: false },
            wordWrap: "on",
            scrollBeyondLastLine: false,
            automaticLayout: true,
            tabSize: 2,
            padding: { top: 8 },
            readOnly: readOnly,
            renderWhitespace: "selection",
            bracketPairColorization: { enabled: true },
            quickSuggestions: true,
            snippetSuggestions: "inline",
          }}
        />
      </Suspense>
    </div>
  );
}
