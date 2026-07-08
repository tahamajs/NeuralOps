import { useState } from "react";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Input } from "../../../components/ui/Input";
import { Select } from "../../../components/ui/Select";
import { Button } from "../../../components/ui/Button";
import { X } from "lucide-react";
import { FilterOptions, ExpertiseType } from "../types";

interface FilterBarProps {
  filters: FilterOptions;
  onFilterChange: (filters: FilterOptions) => void;
}

const expertiseOptions: ExpertiseType[] = [
  "Computer Vision",
  "NLP",
  "Reinforcement Learning",
  "Robotics",
  "HCI",
  "Data Science",
  "Bioinformatics",
  "Quantum Computing",
];

export function FilterBar({ filters, onFilterChange }: FilterBarProps) {
  const [localFilters, setLocalFilters] = useState(filters);

  const handleExpertiseToggle = (tag: ExpertiseType) => {
    const current = localFilters.expertise || [];
    const updated = current.includes(tag)
      ? current.filter((t) => t !== tag)
      : [...current, tag];
    setLocalFilters({ ...localFilters, expertise: updated });
  };

  const applyFilters = () => onFilterChange(localFilters);
  const resetFilters = () => {
    setLocalFilters({});
    onFilterChange({});
  };

  return (
    <TerminalCard className="p-4 space-y-3">
      <div className="grid md:grid-cols-3 gap-3">
        <Select
          label="Institution"
          options={[
            { value: "", label: "All" },
            { value: "Politecnico", label: "Politecnico" },
            { value: "MIT", label: "MIT" },
            { value: "Stanford", label: "Stanford" },
          ]}
          value={localFilters.institution || ""}
          onChange={(e) =>
            setLocalFilters({ ...localFilters, institution: e.target.value })
          }
        />
        <Select
          label="Availability"
          options={[
            { value: "", label: "All" },
            { value: "available", label: "Available" },
            { value: "busy", label: "Busy" },
            { value: "looking", label: "Looking" },
          ]}
          value={localFilters.availability || ""}
          onChange={(e) =>
            setLocalFilters({
              ...localFilters,
              availability: e.target.value as any,
            })
          }
        />
        <Select
          label="Sort by"
          options={[
            { value: "name", label: "Name" },
            { value: "reputation", label: "Reputation" },
            { value: "recent", label: "Most Recent" },
          ]}
          value={localFilters.sort_by || "name"}
          onChange={(e) =>
            setLocalFilters({ ...localFilters, sort_by: e.target.value as any })
          }
        />
      </div>
      <div>
        <label className="text-xs text-slate-400 mb-1 block">Expertise</label>
        <div className="flex flex-wrap gap-2">
          {expertiseOptions.map((tag) => (
            <button
              key={tag}
              onClick={() => handleExpertiseToggle(tag)}
              className={`px-2 py-1 rounded text-xs font-medium transition-colors ${(localFilters.expertise || []).includes(tag) ? "bg-neon-blue text-white" : "bg-white/10 text-slate-300 hover:bg-white/20"}`}
            >
              {tag}
            </button>
          ))}
        </div>
      </div>
      <div className="flex gap-2">
        <Button size="sm" onClick={applyFilters}>
          Apply Filters
        </Button>
        <Button size="sm" variant="ghost" onClick={resetFilters}>
          <X size={14} className="mr-1" /> Reset
        </Button>
      </div>
    </TerminalCard>
  );
}
