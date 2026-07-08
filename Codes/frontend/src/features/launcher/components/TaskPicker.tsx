import { useQuery } from "@tanstack/react-query";
import api from "../../../api/client";
import { ENDPOINTS } from "../../../api/endpoints";
import { Select } from "../../../components/ui/Select";

export const TaskPicker = ({
  value,
  onChange,
}: {
  value: string;
  onChange: (v: string) => void;
}) => {
  const { data } = useQuery({
    queryKey: ["config/tasks"],
    queryFn: async () => (await api.get(ENDPOINTS.CONFIG.TASKS)).data,
  });
  const tasks = data?.tasks || [];
  return (
    <Select
      label="Dataset Task"
      options={tasks.map((t: string) => ({ value: t, label: t.toUpperCase() }))}
      value={value}
      onChange={(e) => onChange(e.target.value)}
    />
  );
};
