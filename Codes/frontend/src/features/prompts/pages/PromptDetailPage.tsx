import { useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import api from "../../../api/client";
import { Card } from "../../../components/ui/Card";
import { Spinner } from "../../../components/ui/Spinner";
import { Button } from "../../../components/ui/Button";
import { useState } from "react";

export default function PromptDetailPage() {
  const { name } = useParams<{ name: string }>();
  const [showHistory, setShowHistory] = useState(false);

  const { data: prompt, isLoading } = useQuery({
    queryKey: ["prompt", name],
    queryFn: async () => (await api.get(`/prompts/${name}/`)).data,
  });

  const { data: history } = useQuery({
    queryKey: ["promptHistory", name],
    queryFn: async () => (await api.get(`/prompts/${name}/history/`)).data,
    enabled: showHistory,
  });

  if (isLoading) return <Spinner />;

  return (
    <div className="space-y-6 animate-fade-in max-w-3xl mx-auto">
      <h1 className="text-2xl font-black text-white">
        PROMPT: <span className="text-neon-blue">{name}</span>
      </h1>
      <Card>
        <h3 className="text-sm font-bold text-white mb-2">Template</h3>
        <pre className="bg-cyber-800 p-4 rounded-lg text-slate-300 text-sm whitespace-pre-wrap">
          {prompt?.template}
        </pre>
      </Card>
      <Button variant="secondary" onClick={() => setShowHistory(!showHistory)}>
        {showHistory ? "Hide History" : "Show History"}
      </Button>
      {showHistory && history && (
        <Card>
          <h3 className="text-sm font-bold text-white mb-4">Version History</h3>
          <ul className="space-y-2">
            {history.map((item: any, idx: number) => (
              <li
                key={idx}
                className="text-xs text-slate-400 border-b border-white/5 pb-2"
              >
                <span className="font-mono text-neon-blue">
                  v{item.version}
                </span>{" "}
                – {item.timestamp}
              </li>
            ))}
          </ul>
        </Card>
      )}
    </div>
  );
}
