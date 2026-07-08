import { useState } from "react";
import { useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import api from "../../../api/client";
import { Button } from "../../../components/ui/Button";
import { Card } from "../../../components/ui/Card";
import { Select } from "../../../components/ui/Select";
import { Spinner } from "../../../components/ui/Spinner";

export default function ModelVersionComparePage() {
  const { runId } = useParams<{ runId: string }>();
  const [va, setVa] = useState("");
  const [vb, setVb] = useState("");
  const [diff, setDiff] = useState<any>(null);

  const { data: versions = [] } = useQuery({
    queryKey: ["modelVersions", runId],
    queryFn: async () =>
      (await api.get(`/api/model-versioning/runs/${runId}/versions/`)).data,
  });

  const handleCompare = async () => {
    if (!va || !vb) return;
    const res = await api.get(
      `/api/model-versioning/runs/${runId}/versions/compare/${va}/${vb}/`,
    );
    setDiff(res.data);
  };

  const options = versions.map((v: any) => ({
    value: v.version_number.toString(),
    label: v.version,
  }));

  return (
    <div className="space-y-6 animate-fade-in">
      <h1 className="text-2xl font-black text-white">Compare Model Versions</h1>
      <Card>
        <div className="flex gap-4 mb-4">
          <Select
            label="Version A"
            options={options}
            value={va}
            onChange={(e) => setVa(e.target.value)}
          />
          <Select
            label="Version B"
            options={options}
            value={vb}
            onChange={(e) => setVb(e.target.value)}
          />
          <Button
            variant="primary"
            onClick={handleCompare}
            className="self-end"
          >
            Compare
          </Button>
        </div>
        {diff && (
          <pre className="bg-white/5 p-4 rounded-lg text-xs text-slate-400 overflow-auto max-h-96">
            {JSON.stringify(diff, null, 2)}
          </pre>
        )}
      </Card>
    </div>
  );
}
