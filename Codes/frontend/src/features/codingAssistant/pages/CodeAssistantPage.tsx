import { useState, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { codingAssistantAPI } from "../services/api";
import { useCodingAssistantWebSocket } from "../hooks/useCodingAssistantWebSocket";
import { ChatPanel } from "../components/ChatPanel";
import { FileBrowser } from "../components/FileBrowser";
import { CodeEditorPanel } from "../components/CodeEditorPanel";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Spinner } from "../../../components/ui/Spinner";
import { Tabs, TabList, Tab, TabPanel } from "../../../components/ui/Tabs";
import { Message, GeneratedFile } from "../types";

export default function CodeAssistantPage({
  onClose,
}: {
  onClose?: () => void;
}) {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [currentFileId, setCurrentFileId] = useState<string | null>(null);
  const [currentFileContent, setCurrentFileContent] = useState("");
  const [currentFileLanguage, setCurrentFileLanguage] = useState("python");
  const [isSaving, setIsSaving] = useState(false);
  const queryClient = useQueryClient();

  const { data: sessionData, isLoading: sessionLoading } = useQuery({
    queryKey: ["code-session", sessionId],
    queryFn: () => codingAssistantAPI.getSession(sessionId!),
    enabled: !!sessionId,
  });

  const { data: files = [], refetch: refetchFiles } = useQuery({
    queryKey: ["code-files", sessionId],
    queryFn: () => codingAssistantAPI.listFiles(sessionId!),
    enabled: !!sessionId,
  });

  const { data: fileContent } = useQuery({
    queryKey: ["code-file", currentFileId],
    queryFn: () => codingAssistantAPI.getFileContent(currentFileId!),
    enabled: !!currentFileId,
    onSuccess: (data) => setCurrentFileContent(data.content),
  });

  const createSession = useMutation({
    mutationFn: () => codingAssistantAPI.createSession(),
    onSuccess: (res) => setSessionId(res.data.session_id),
  });
  const saveFile = useMutation({
    mutationFn: (data: { fileId: string; content: string }) =>
      codingAssistantAPI.saveFileContent(data.fileId, data.content),
    onSuccess: () => setIsSaving(false),
  });
  const deleteFile = useMutation({
    mutationFn: (fileId: string) => codingAssistantAPI.deleteFile(fileId),
    onSuccess: () => refetchFiles(),
  });
  const generateConfig = useMutation({
    mutationFn: (params: any) =>
      codingAssistantAPI.generateExperimentConfig(params),
  });

  const {
    isConnected,
    streamingContent,
    suggestions,
    sendMessage,
    applySuggestion,
  } = useCodingAssistantWebSocket(sessionId);

  useEffect(() => {
    createSession.mutate();
  }, []);
  useEffect(() => {
    if (files[0] && !currentFileId) setCurrentFileId(files[0].id);
  }, [files]);

  const handleSendMessage = (content: string) => {
    setMessages((prev) => [
      ...prev,
      {
        id: Date.now().toString(),
        role: "user",
        content,
        timestamp: new Date(),
        suggestions: [],
      },
    ]);
    sendMessage(content);
  };

  const handleSaveFile = async (content: string) => {
    if (currentFileId) {
      setIsSaving(true);
      await saveFile.mutateAsync({ fileId: currentFileId, content });
    }
  };

  const handleCreateFile = (name: string) => {
    codingAssistantAPI.createFile(sessionId!, name).then(() => refetchFiles());
  };
  const handleDeleteFile = (id: string) => {
    deleteFile.mutate(id);
    if (currentFileId === id) setCurrentFileId(null);
  };

  if (sessionLoading) return <Spinner />;

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <h1 className="text-xl font-bold text-white flex items-center gap-2">
          <span className="text-neon-mint">&lt;/&gt;</span> AI Coding Assistant
        </h1>
        {onClose && (
          <Button variant="ghost" size="sm" onClick={onClose}>
            ✕
          </Button>
        )}
      </div>
      <div className="flex-1 overflow-hidden p-4">
        <div className="grid lg:grid-cols-12 gap-4 h-full">
          <div className="lg:col-span-3 h-full">
            <FileBrowser
              files={files}
              currentFileId={currentFileId}
              onSelectFile={setCurrentFileId}
              onDeleteFile={handleDeleteFile}
              onCreateFile={handleCreateFile}
            />
          </div>
          <div className="lg:col-span-6 h-full">
            <CodeEditorPanel
              content={currentFileContent}
              language={currentFileLanguage}
              onSave={handleSaveFile}
              isSaving={isSaving}
            />
          </div>
          <div className="lg:col-span-3 h-full">
            <ChatPanel
              messages={messages}
              onSendMessage={handleSendMessage}
              isStreaming={false}
              streamingContent={streamingContent}
              suggestions={suggestions}
              onApplySuggestion={applySuggestion}
              isConnected={isConnected}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
