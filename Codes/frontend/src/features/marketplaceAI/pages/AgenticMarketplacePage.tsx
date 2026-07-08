import { useState, useEffect, useRef } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { marketplaceAIApi } from "../services/api";
import { useMarketplaceAgentWebSocket } from "../hooks/useMarketplaceAgentWebSocket";
import { AgentChatPanel } from "../components/AgentChatPanel";
import { ListingCard } from "../components/ListingCard";
import { SuggestionChips } from "../components/SuggestionChips";
import { ComplianceCheckWidget } from "../components/ComplianceCheckWidget";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";
import { Spinner } from "../../../components/ui/Spinner";
import { Tabs, TabList, Tab, TabPanel } from "../../../components/ui/Tabs";
import { MessageSquare, Cpu, Shield, Plus } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

export default function AgenticMarketplacePage({
  onClose,
}: {
  onClose?: () => void;
}) {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState("chat");
  const [filters, setFilters] = useState({
    search: "",
    minPrice: "",
    maxPrice: "",
    region: "",
  });
  const { success, error: toastError } = useToast();

  const startSession = useMutation({
    mutationFn: () => marketplaceAIApi.startAgentSession(),
    onSuccess: (res) => setSessionId(res.data.session_id),
  });

  const {
    data: listingsData,
    isLoading: listingsLoading,
    refetch,
  } = useQuery({
    queryKey: ["marketplace-listings", filters],
    queryFn: () => marketplaceAIApi.getListings(filters),
    enabled: !!sessionId,
  });

  const {
    isConnected,
    streamingContent,
    liveListings,
    suggestions,
    complianceUpdates,
    sendMessage,
    requestCompliance,
    applyFilter,
  } = useMarketplaceAgentWebSocket(sessionId);

  useEffect(() => {
    startSession.mutate();
  }, []);

  const displayedListings = liveListings.length
    ? liveListings
    : listingsData?.results || [];

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <h1 className="text-xl font-bold text-white flex items-center gap-2">
          <MessageSquare size={20} className="text-neon-mint" /> AI Agent
          Marketplace
        </h1>
        <div className="flex gap-2">
          <span
            className={`text-xs ${isConnected ? "text-green-400" : "text-yellow-400"}`}
          >
            {isConnected ? "● Agent active" : "○ Connecting..."}
          </span>
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>
      <div className="flex-1 overflow-hidden p-4">
        <div className="grid lg:grid-cols-3 gap-6 h-full">
          <div className="lg:col-span-2 flex flex-col gap-4">
            <div className="flex gap-2">
              <Input
                placeholder="Search listings..."
                value={filters.search}
                onChange={(e) =>
                  setFilters({ ...filters, search: e.target.value })
                }
                className="flex-1"
              />
              <Button size="sm" variant="ghost" onClick={() => refetch()}>
                Search
              </Button>
            </div>
            <Tabs value={activeTab} onValueChange={setActiveTab}>
              <TabList>
                <Tab value="chat">AI Chat</Tab>
                <Tab value="listings">Listings</Tab>
                <Tab value="compliance">Compliance</Tab>
              </TabList>
              <TabPanel value="chat">
                <AgentChatPanel
                  onSendMessage={sendMessage}
                  isConnected={isConnected}
                  streamingContent={streamingContent}
                  suggestions={suggestions}
                />
              </TabPanel>
              <TabPanel value="listings">
                <div className="space-y-3 max-h-[500px] overflow-y-auto pr-2">
                  {listingsLoading ? (
                    <Spinner />
                  ) : displayedListings.length ? (
                    displayedListings.map((listing) => (
                      <ListingCard
                        key={listing.id}
                        listing={listing}
                        onCheckCompliance={() => requestCompliance(listing.id)}
                      />
                    ))
                  ) : (
                    <div className="text-center text-slate-500 py-8">
                      No listings found. Try asking the AI agent.
                    </div>
                  )}
                </div>
              </TabPanel>
              <TabPanel value="compliance">
                <ComplianceCheckWidget complianceUpdates={complianceUpdates} />
              </TabPanel>
            </Tabs>
          </div>
          <div className="space-y-4">
            {suggestions.length > 0 && (
              <SuggestionChips
                suggestions={suggestions}
                onSelect={sendMessage}
              />
            )}
            <TerminalCard className="p-3">
              <div className="text-xs text-slate-400 mb-1">Active filters</div>
              <div className="flex flex-wrap gap-2">
                {filters.minPrice && (
                  <span className="px-2 py-1 rounded bg-white/10 text-xs">
                    Min ${filters.minPrice}
                  </span>
                )}
                {filters.maxPrice && (
                  <span className="px-2 py-1 rounded bg-white/10 text-xs">
                    Max ${filters.maxPrice}
                  </span>
                )}
                {filters.region && (
                  <span className="px-2 py-1 rounded bg-white/10 text-xs">
                    Region: {filters.region}
                  </span>
                )}
                {!filters.minPrice && !filters.maxPrice && !filters.region && (
                  <span className="text-slate-500">No active filters</span>
                )}
              </div>
            </TerminalCard>
          </div>
        </div>
      </div>
    </div>
  );
}
