import { Badge } from "./Badge";
export const StatusBadge = ({ status }: { status: string }) => (
  <Badge variant={status as any}>{status}</Badge>
);
