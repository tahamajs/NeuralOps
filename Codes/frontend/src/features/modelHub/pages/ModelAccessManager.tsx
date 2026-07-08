// src/features/modelHub/pages/ModelAccessManager.tsx
import { useState } from "react";
import { useParams } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { modelHubAPI } from "../services/api";
import { AccessPolicyForm } from "../components/AccessPolicyForm";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";
import { Select } from "../../../components/ui/Select";
import { Spinner } from "../../../components/ui/Spinner";
import { DataGrid } from "../../../components/ui/DataGrid";
import { Shield, Plus, Trash2 } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

export default function ModelAccessManager({
  onClose,
}: {
  onClose?: () => void;
}) {
  const { modelId } = useParams();
  const [showForm, setShowForm] = useState(false);
  const queryClient = useQueryClient();
  const { success, error } = useToast();

  const { data: policies, isLoading } = useQuery({
    queryKey: ["access-policies", modelId],
    queryFn: () => modelHubAPI.getAccessPolicies(modelId!),
    enabled: !!modelId,
  });

  const revokeAccess = useMutation({
    mutationFn: (policyId: string) => modelHubAPI.revokeAccess(policyId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["access-policies", modelId] });
      success("Access revoked");
    },
  });

  const columns = [
    { accessorKey: "username", header: "User" },
    {
      accessorKey: "permission",
      header: "Permission",
      cell: ({ row }: any) => (
        <Badge variant="info">{row.original.permission}</Badge>
      ),
    },
    { accessorKey: "granted_by", header: "Granted By" },
    {
      accessorKey: "granted_at",
      header: "Granted At",
      cell: ({ row }: any) =>
        new Date(row.original.granted_at).toLocaleString(),
    },
    {
      accessorKey: "expires_at",
      header: "Expires",
      cell: ({ row }: any) =>
        row.original.expires_at
          ? new Date(row.original.expires_at).toLocaleDateString()
          : "Never",
    },
    {
      id: "actions",
      header: "",
      cell: ({ row }: any) => (
        <button onClick={() => revokeAccess.mutate(row.original.id)}>
          <Trash2 size={14} className="text-red-400" />
        </button>
      ),
    },
  ];

  if (isLoading) return <Spinner />;

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <Shield size={20} className="text-neon-mint" /> Model Access Manager
          </h1>
          <p className="text-xs text-slate-400">
            Manage user permissions for model {modelId?.slice(0, 8)}
          </p>
        </div>
        <div className="flex gap-2">
          <Button size="sm" onClick={() => setShowForm(true)}>
            <Plus size={14} className="mr-1" /> Grant Access
          </Button>
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>
      <div className="flex-1 overflow-auto p-4 space-y-4">
        <TerminalCard noPadding>
          <DataGrid data={policies || []} columns={columns} />
        </TerminalCard>
        <AccessPolicyForm
          open={showForm}
          onClose={() => setShowForm(false)}
          modelId={modelId!}
          onSuccess={() =>
            queryClient.invalidateQueries({
              queryKey: ["access-policies", modelId],
            })
          }
        />
      </div>
    </div>
  );
}
