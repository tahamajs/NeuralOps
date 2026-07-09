import { useParams } from "react-router-dom";
import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import api from "../../../api/client";
import { Button } from "../../../components/ui/Button";
import { Card } from "../../../components/ui/Card";
import { Select } from "../../../components/ui/Select";
import { useQuery } from "@tanstack/react-query";

export default function SetProductionVersionPage() {
  const { runId } = useParams<{ runId: string }>();
  const [version, setVersion] = useState("");
  const { data: versions = [] } = useQuery({
    queryKey: ["modelVersions", runId],
    queryFn: async () =>
      (await api.get(`/model-versioning/runs/${runId}/versions/`)).data,
  });
  const mutation = useMutation({
    mutationFn: () =>
      api.post(
        `/model-versioning/runs/${runId}/versions/${version}/set_production/`,
      ),
    onSuccess: () => alert("Production version set"),
  });

  const options = versions.map((v: any) => ({
    value: v.version_number.toString(),
    label: v.version,
  }));

  return (
    <div className="space-y-6 animate-fade-in max-w-xl mx-auto">
      <h1 className="text-2xl font-black text-white">
        SET PRODUCTION <span className="text-neon-blue">VERSION</span>
      </h1>
      <Card>
        <Select
          label="Choose Version"
          options={options}
          value={version}
          onChange={(e) => setVersion(e.target.value)}
        />
        <Button
          className="mt-4"
          onClick={() => mutation.mutate()}
          disabled={mutation.isPending || !version}
        >
          Set as Production
        </Button>
      </Card>
    </div>
  );
}
