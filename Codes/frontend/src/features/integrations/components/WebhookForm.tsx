import { useState } from "react";
import { Input } from "../../../components/ui/Input";
import { Select } from "../../../components/ui/Select";
import { Button } from "../../../components/ui/Button";
import { useToast } from "../../../components/ui/Toast";
import { integrationsAPI } from "../services/api";
import { Webhook, WebhookEvent } from "../types";

const EVENT_OPTIONS = [
  { value: "experiment.created", label: "Experiment Created" },
  { value: "experiment.completed", label: "Experiment Completed" },
  { value: "run.failed", label: "Run Failed" },
  { value: "model.deployed", label: "Model Deployed" },
];

export function WebhookForm({
  initialData,
  onSuccess,
}: {
  initialData?: Webhook;
  onSuccess: () => void;
}) {
  const [name, setName] = useState(initialData?.name || "");
  const [url, setUrl] = useState(initialData?.url || "");
  const [events, setEvents] = useState<WebhookEvent[]>(
    initialData?.events || [],
  );
  const [secret, setSecret] = useState(initialData?.secret || "");
  const [loading, setLoading] = useState(false);
  const { success, error } = useToast();

  const handleSubmit = async () => {
    setLoading(true);
    try {
      if (initialData)
        await integrationsAPI.updateWebhook(initialData.id, {
          name,
          url,
          events,
          secret,
        });
      else await integrationsAPI.createWebhook({ name, url, events, secret });
      success(initialData ? "Webhook updated" : "Webhook created");
      onSuccess();
    } catch {
      error("Failed to save webhook");
    } finally {
      setLoading(false);
    }
  };

  const toggleEvent = (event: WebhookEvent) =>
    setEvents((prev) =>
      prev.includes(event) ? prev.filter((e) => e !== event) : [...prev, event],
    );

  return (
    <div className="space-y-4">
      <Input
        label="Name"
        value={name}
        onChange={(e) => setName(e.target.value)}
      />
      <Input
        label="Endpoint URL"
        value={url}
        onChange={(e) => setUrl(e.target.value)}
      />
      <div>
        <label className="text-xs text-slate-400 block mb-2">Events</label>
        <div className="flex flex-wrap gap-2">
          {EVENT_OPTIONS.map((opt) => (
            <button
              key={opt.value}
              type="button"
              onClick={() => toggleEvent(opt.value as WebhookEvent)}
              className={`px-3 py-1 rounded text-sm ${events.includes(opt.value as WebhookEvent) ? "bg-neon-blue text-white" : "bg-white/10 text-slate-300"}`}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>
      <Input
        label="Secret (optional)"
        value={secret}
        onChange={(e) => setSecret(e.target.value)}
        type="password"
      />
      <Button onClick={handleSubmit} disabled={loading} className="w-full">
        {loading ? "Saving..." : initialData ? "Update" : "Create"}
      </Button>
    </div>
  );
}
