import { Modal } from "../../../components/ui/Modal";
import { Button } from "../../../components/ui/Button";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Plugin } from "../types";

export function PluginInstallModal({
  plugin,
  onClose,
  onConfirm,
}: {
  plugin: Plugin;
  onClose: () => void;
  onConfirm: () => void;
}) {
  return (
    <Modal open={!!plugin} onClose={onClose} title={`Install ${plugin?.name}`}>
      <TerminalCard className="p-4 space-y-4">
        <p>
          The plugin <strong>{plugin?.name}</strong> will have access to:
        </p>
        <ul className="list-disc list-inside text-sm text-slate-300">
          {plugin?.permissions.map((p) => (
            <li key={p}>{p}</li>
          ))}
        </ul>
        <div className="flex gap-2">
          <Button onClick={onConfirm}>Confirm Installation</Button>
          <Button variant="ghost" onClick={onClose}>
            Cancel
          </Button>
        </div>
      </TerminalCard>
    </Modal>
  );
}
