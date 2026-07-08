// src/features/integrations/pages/PluginMarketplace.tsx
import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { integrationsAPI } from "../services/api";
import { PluginCard } from "../components/PluginCard";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";
import { Select } from "../../../components/ui/Select";
import { Spinner } from "../../../components/ui/Spinner";
import { Tabs, TabList, Tab, TabPanel } from "../../../components/ui/Tabs";
import { Puzzle, Search, Download, Trash2, Settings } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

export default function PluginMarketplace({
  onClose,
}: {
  onClose?: () => void;
}) {
  const [search, setSearch] = useState("");
  const [activeTab, setActiveTab] = useState("available");
  const queryClient = useQueryClient();
  const { success, error } = useToast();

  const { data: plugins, isLoading } = useQuery({
    queryKey: ["plugins"],
    queryFn: () => integrationsAPI.getPlugins(),
  });

  const installPlugin = useMutation({
    mutationFn: (id: string) => integrationsAPI.installPlugin(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["plugins"] });
      success("Plugin installed");
    },
  });

  const uninstallPlugin = useMutation({
    mutationFn: (id: string) => integrationsAPI.uninstallPlugin(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["plugins"] });
      success("Plugin uninstalled");
    },
  });

  if (isLoading) return <Spinner />;

  const installedPlugins = plugins?.filter((p) => p.installed) || [];
  const availablePlugins = plugins?.filter((p) => !p.installed) || [];
  const filteredPlugins = (
    activeTab === "installed" ? installedPlugins : availablePlugins
  ).filter((p) => p.name.toLowerCase().includes(search.toLowerCase()));

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <Puzzle size={20} className="text-neon-mint" /> Plugin Marketplace
          </h1>
          <p className="text-xs text-slate-400">
            Extend platform functionality with plugins
          </p>
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
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <div className="flex justify-between items-center mb-4">
            <TabList>
              <Tab value="available">Available</Tab>
              <Tab value="installed">Installed ({installedPlugins.length})</Tab>
            </TabList>
            <div className="relative">
              <Search
                size={14}
                className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400"
              />
              <Input
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search plugins..."
                className="pl-9 w-64"
              />
            </div>
          </div>
          <TabPanel value="available">
            <div className="grid md:grid-cols-2 gap-4">
              {filteredPlugins.map((plugin) => (
                <PluginCard
                  key={plugin.id}
                  plugin={plugin}
                  onInstall={() => installPlugin.mutate(plugin.id)}
                />
              ))}
            </div>
          </TabPanel>
          <TabPanel value="installed">
            <div className="grid md:grid-cols-2 gap-4">
              {filteredPlugins.map((plugin) => (
                <PluginCard
                  key={plugin.id}
                  plugin={plugin}
                  onUninstall={() => uninstallPlugin.mutate(plugin.id)}
                  installed
                />
              ))}
            </div>
          </TabPanel>
        </Tabs>
      </div>
    </div>
  );
}
