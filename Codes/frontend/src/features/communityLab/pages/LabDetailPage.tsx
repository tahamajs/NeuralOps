// src/features/communityLab/pages/LabDetailPage.tsx
import { useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { communityLabAPI } from "../services/api";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Spinner } from "../../../components/ui/Spinner";
import { Badge } from "../../../components/ui/Badge";
import { Tabs, TabList, Tab, TabPanel } from "../../../components/ui/Tabs";
import { AgentRoleManager } from "../components/AgentRoleManager";
import { ChatroomView } from "../components/ChatroomView";
import { WorkflowBuilder } from "../components/WorkflowBuilder";
import { DiscoveryFeed } from "../components/DiscoveryFeed";
import { PeerReviewPanel } from "../components/PeerReviewPanel";
import {
  Users,
  Bot,
  GitBranch,
  MessageSquare,
  Compass,
  ArrowLeft,
} from "lucide-react";

export default function LabDetailPage({ onClose }: { onClose?: () => void }) {
  const { labId } = useParams();
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState("overview");

  const { data: lab, isLoading } = useQuery({
    queryKey: ["lab", labId],
    queryFn: () => communityLabAPI.getLab(labId!),
    enabled: !!labId,
  });

  if (isLoading) return <Spinner />;
  if (!lab) return <div className="p-4 text-red-400">Lab not found</div>;

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div className="flex items-center gap-3">
          <Button variant="ghost" size="sm" onClick={() => navigate("/labs")}>
            <ArrowLeft size={16} />
          </Button>
          <div>
            <h1 className="text-xl font-bold text-white">{lab.name}</h1>
            <p className="text-xs text-slate-400">
              {lab.description || "No description"}
            </p>
          </div>
        </div>
        <div className="flex gap-2">
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>
      <div className="flex-1 overflow-auto p-4">
        <div className="grid lg:grid-cols-3 gap-4 mb-6">
          <TerminalCard className="p-3 text-center">
            <Users className="mx-auto text-neon-blue mb-1" />
            <div className="text-2xl font-bold text-white">
              {lab.member_count}
            </div>
            <div className="text-xs text-slate-400">Members</div>
          </TerminalCard>
          <TerminalCard className="p-3 text-center">
            <Bot className="mx-auto text-neon-mint mb-1" />
            <div className="text-2xl font-bold text-white">
              {lab.agent_count}
            </div>
            <div className="text-xs text-slate-400">Agents</div>
          </TerminalCard>
          <TerminalCard className="p-3 text-center">
            <GitBranch className="mx-auto text-yellow-400 mb-1" />
            <div className="text-2xl font-bold text-white">
              {lab.workflow_count}
            </div>
            <div className="text-xs text-slate-400">Workflows</div>
          </TerminalCard>
        </div>
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabList>
            <Tab value="agents">
              <Bot size={14} className="mr-1" /> Agents
            </Tab>
            <Tab value="chat">
              <MessageSquare size={14} className="mr-1" /> Chatrooms
            </Tab>
            <Tab value="workflows">
              <GitBranch size={14} className="mr-1" /> Workflows
            </Tab>
            <Tab value="discoveries">
              <Compass size={14} className="mr-1" /> Discoveries
            </Tab>
            <Tab value="reviews">
              <Compass size={14} className="mr-1" /> Peer Review
            </Tab>
          </TabList>
          <TabPanel value="agents">
            <AgentRoleManager labId={labId!} />
          </TabPanel>
          <TabPanel value="chat">
            <ChatroomView labId={labId!} />
          </TabPanel>
          <TabPanel value="workflows">
            <WorkflowBuilder labId={labId!} />
          </TabPanel>
          <TabPanel value="discoveries">
            <DiscoveryFeed labId={labId!} />
          </TabPanel>
          <TabPanel value="reviews">
            <PeerReviewPanel labId={labId!} />
          </TabPanel>
        </Tabs>
      </div>
    </div>
  );
}
