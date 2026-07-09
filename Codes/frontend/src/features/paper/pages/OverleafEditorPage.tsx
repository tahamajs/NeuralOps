import { useState, useEffect, useCallback, useRef } from "react";
import { useParams } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import api from "../../../api/client";
import { ENDPOINTS } from "../../../api/endpoints";
import FileTreeSidebar from "../components/FileTreeSidebar";
import { EditorPanel } from "../components/EditorPanel";
import PDFPreviewPanel from "../components/PDFPreviewPanel";
import EditorToolbar from "../components/EditorToolbar";
import CompileErrorsPanel from "../components/CompileErrorsPanel";
import { Card } from "../../../components/ui/Card";
import { Spinner } from "../../../components/ui/Spinner";
import { useFloatingWindowStore } from "../../../components/floating/FloatingWindowManager";

export default function OverleafEditorPage() {
  const { projectId } = useParams<{ projectId?: string }>();
  const queryClient = useQueryClient();
  const [selectedFile, setSelectedFile] = useState<string>("main.tex");
  const [fileContent, setFileContent] = useState("");
  const [pdfUrl, setPdfUrl] = useState<string | null>(null);
  const [compiling, setCompiling] = useState(false);
  const [compileLog, setCompileLog] = useState("");
  const [errors, setErrors] = useState<any[]>([]);
  const [showErrors, setShowErrors] = useState(false);
  const [autoCompile, setAutoCompile] = useState(false);
  const editorRef = useRef<any>(null);

  // Fetch project data (files list)
  const { data: project, isLoading } = useQuery({
    queryKey: ["paperProject", projectId],
    queryFn: async () =>
      (await api.get(`/paper/projects/${projectId}/`)).data,
    enabled: !!projectId,
  });

  // Fetch file content
  const { data: fileData } = useQuery({
    queryKey: ["paperFile", projectId, selectedFile],
    queryFn: async () => {
      const res = await api.get(
        `/paper/projects/${projectId}/files/${selectedFile}/`,
      );
      return res.data;
    },
    enabled: !!projectId && !!selectedFile,
  });

  useEffect(() => {
    if (fileData) {
      setFileContent(fileData.content || "");
    }
  }, [fileData]);

  // Compile mutation
  const compileMut = useMutation({
    mutationFn: async () => {
      setCompiling(true);
      setShowErrors(false);
      const res = await api.post(`/paper/projects/${projectId}/compile/`);
      return res.data;
    },
    onSuccess: (data) => {
      setCompiling(false);
      if (data.status === "completed" && data.pdf_url) {
        setPdfUrl(data.pdf_url);
        setErrors([]);
      } else {
        setErrors(data.log || "Compilation failed");
        setShowErrors(true);
      }
    },
    onError: (err: any) => {
      setCompiling(false);
      setErrors([err.message]);
      setShowErrors(true);
    },
  });

  // Auto-compile debounce
  useEffect(() => {
    if (!autoCompile) return;
    const timer = setTimeout(() => {
      compileMut.mutate();
    }, 1500);
    return () => clearTimeout(timer);
  }, [fileContent, autoCompile]);

  const handleFileSelect = (filename: string) => {
    setSelectedFile(filename);
  };

  const handleContentChange = (content: string) => {
    setFileContent(content);
    // Auto-save via PATCH
    // We could debounce this to avoid too many requests
  };

  const handleSave = async () => {
    if (!projectId || !selectedFile) return;
    await api.put(`/paper/projects/${projectId}/files/${selectedFile}/`, {
      content: fileContent,
    });
    queryClient.invalidateQueries({ queryKey: ["paperProject", projectId] });
  };

  const handleDownloadPDF = () => {
    if (pdfUrl) {
      window.open(pdfUrl, "_blank");
    }
  };

  if (isLoading) return <Spinner />;

  return (
    <div className="h-[calc(100vh-57px)] flex flex-col bg-cyber-900 text-white">
      {/* Top Toolbar */}
      <EditorToolbar
        projectId={projectId}
        compiling={compiling}
        onCompile={() => compileMut.mutate()}
        onDownloadPDF={handleDownloadPDF}
        onSave={handleSave}
        autoCompile={autoCompile}
        onToggleAutoCompile={() => setAutoCompile(!autoCompile)}
        pdfUrl={pdfUrl}
      />

      {/* Error Panel */}
      {showErrors && errors.length > 0 && (
        <CompileErrorsPanel
          errors={errors}
          onClose={() => setShowErrors(false)}
        />
      )}

      {/* Main Split */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Sidebar – File Tree */}
        <div className="w-60 border-r border-white/5 flex-shrink-0 overflow-y-auto bg-cyber-800">
          <FileTreeSidebar
            projectId={projectId}
            selectedFile={selectedFile}
            onFileSelect={handleFileSelect}
            files={project?.files || []}
            onRefresh={() =>
              queryClient.invalidateQueries({
                queryKey: ["paperProject", projectId],
              })
            }
          />
        </div>

        {/* Editor + Preview Split */}
        <div className="flex-1 flex overflow-hidden">
          {/* Editor Pane */}
          <div className="flex-1 border-r border-white/5">
            <EditorPanel
              content={fileContent}
              onChange={handleContentChange}
              language="latex"
              onSave={handleSave}
            />
          </div>

          {/* PDF Preview Pane */}
          <div className="flex-1 bg-slate-100">
            <PDFPreviewPanel pdfUrl={pdfUrl} compiling={compiling} />
          </div>
        </div>
      </div>
    </div>
  );
}
