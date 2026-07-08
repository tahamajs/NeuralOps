// src/features/experimentExplorer/pages/SQLQueryEditor.tsx
import { useState, useEffect, useRef, useCallback } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import Editor from "@monaco-editor/react";
import { experimentExplorerAPI } from "../services/api";
import { SQLResultTable } from "../components/SQLResultTable";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Select } from "../../../components/ui/Select";
import { Spinner } from "../../../components/ui/Spinner";
import { Database, Play, Save, Trash2, History, KeyRound } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

export default function SQLQueryEditor({ onClose }: { onClose?: () => void }) {
  const [selectedConnection, setSelectedConnection] = useState("");
  const [query, setQuery] = useState("SELECT * FROM experiments LIMIT 100;");
  const [result, setResult] = useState<any>(null);
  const [isExecuting, setIsExecuting] = useState(false);
  const editorRef = useRef<any>(null);
  const { success, error } = useToast();

  const { data: connections, isLoading: connsLoading } = useQuery({
    queryKey: ["sql-connections"],
    queryFn: () => experimentExplorerAPI.getSQLConnections(),
  });
  const { data: schema } = useQuery({
    queryKey: ["sql-schema", selectedConnection],
    queryFn: () => experimentExplorerAPI.getSchemaInfo(selectedConnection),
    enabled: !!selectedConnection,
  });

  const executeMutation = useMutation({
    mutationFn: () =>
      experimentExplorerAPI.executeSQL(selectedConnection, query),
    onSuccess: (res: any) => {
      setResult(res);
      success(
        "Query executed",
        `${res.rowCount} rows returned in ${res.executionTimeMs}ms`,
      );
    },
    onError: () => error("Query failed", "Please check your syntax"),
  });

  const handleExecute = () => {
    if (selectedConnection && query.trim()) {
      setIsExecuting(true);
      executeMutation.mutate(undefined, {
        onSettled: () => setIsExecuting(false),
      });
    }
  };

  const handleEditorMount = (editor: any) => {
    editorRef.current = editor;
    editor.addCommand(49 | 2048, handleExecute);
  };

  if (connsLoading) return <Spinner />;

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <Database size={20} className="text-neon-mint" /> SQL Query Editor
          </h1>
          <p className="text-xs text-slate-400">
            Run analytics queries against experiment data
          </p>
        </div>
        <div className="flex gap-2">
          <Button size="sm" variant="ghost">
            <History size={14} className="mr-1" /> History
          </Button>
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>
      <div className="flex-1 overflow-hidden p-4">
        <div className="grid lg:grid-cols-4 gap-4 h-full">
          <div className="space-y-4">
            <Select
              options={
                connections?.map((c) => ({ value: c.id, label: c.name })) || []
              }
              value={selectedConnection}
              onChange={(e) => setSelectedConnection(e.target.value)}
              placeholder="Select connection"
            />
            <Button size="sm" variant="ghost">
              <KeyRound size={12} className="mr-1" /> Manage Connections
            </Button>
          </div>
          <div className="lg:col-span-3 flex flex-col gap-4">
            <div className="h-80">
              <Editor
                height="100%"
                language="sql"
                theme="vs-dark"
                value={query}
                onChange={(val) => setQuery(val || "")}
                onMount={handleEditorMount}
                options={{
                  fontSize: 13,
                  minimap: { enabled: false },
                  wordWrap: "on",
                  automaticLayout: true,
                }}
              />
            </div>
            <div className="flex gap-2">
              <Button
                onClick={handleExecute}
                disabled={!selectedConnection || !query.trim() || isExecuting}
              >
                <Play size={14} className="mr-1" /> Execute (Ctrl+Enter)
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setQuery("SELECT * FROM experiments LIMIT 100;")}
              >
                Reset
              </Button>
            </div>
            {executeMutation.isPending && (
              <div className="text-center py-4">
                <Spinner size="sm" />
                <p className="text-slate-400 text-sm mt-1">
                  Executing query...
                </p>
              </div>
            )}
            {result && (
              <div>
                <div className="text-xs text-slate-400 mb-2">
                  {result.rowCount} rows · {result.executionTimeMs}ms
                </div>
                <SQLResultTable columns={result.columns} rows={result.rows} />
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
