import { useQuery } from "@tanstack/react-query";
import api from "../../../api/client";
import { Card } from "../../../components/ui/Card";
import { Spinner } from "../../../components/ui/Spinner";

export default function FederatedStatusPage() {
  const { data, isLoading } = useQuery({
    queryKey: ["federatedStatus"],
    queryFn: async () => (await api.get("/federated/status/")).data,
    refetchInterval: 5000,
  });

  if (isLoading) return <Spinner />;
  return (
    <div className="space-y-6 animate-fade-in">
      <h1 className="text-2xl font-black text-white">
        Federated Learning Status
      </h1>
      <Card>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <div className="text-xs text-slate-500 uppercase tracking-widest">
              Status
            </div>
            <div className="text-xl font-bold text-white">{data.status}</div>
          </div>
          <div>
            <div className="text-xs text-slate-500 uppercase tracking-widest">
              Connected Clients
            </div>
            <div className="text-xl font-bold text-white">
              {data.connected_clients}
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
}
