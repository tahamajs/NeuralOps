import { useQuery } from "@tanstack/react-query";
import api from "../../../api/client";
import { Card } from "../../../components/ui/Card";
import { Spinner } from "../../../components/ui/Spinner";

export default function VersionPage() {
  const { data } = useQuery({
    queryKey: ["version"],
    queryFn: async () => (await api.get("/version/")).data,
  });
  if (!data) return <Spinner />;
  return (
    <div className="space-y-6 animate-fade-in">
      <h1 className="text-2xl font-black text-white">
        API <span className="text-neon-blue">VERSION</span>
      </h1>
      <Card>
        <div className="text-sm text-slate-400 space-y-2">
          <p>
            API Version: <span className="text-white">{data.api_version}</span>
          </p>
          <p>
            Django Version:{" "}
            <span className="text-white">{data.django_version}</span>
          </p>
          <p>
            Environment: <span className="text-white">{data.environment}</span>
          </p>
        </div>
      </Card>
    </div>
  );
}
