// src/features/researchDiscovery/pages/GrantMatcher.tsx
import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { researchDiscoveryAPI } from "../services/api";
import { GrantCard } from "../components/GrantCard";
import { SimilaritySlider } from "../components/SimilaritySlider";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";
import { Select } from "../../../components/ui/Select";
import { Spinner } from "../../../components/ui/Spinner";
import { Target, Award, Users, Briefcase, MapPin } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

export default function GrantMatcher({ onClose }: { onClose?: () => void }) {
  const [profile, setProfile] = useState({
    name: "",
    organization: "",
    role: "",
    country: "",
    research_keywords: "",
    entity_type: "academic",
    min_score: 0.6,
  });
  const [showResults, setShowResults] = useState(false);
  const { success, error } = useToast();

  const matchMutation = useMutation({
    mutationFn: () => researchDiscoveryAPI.matchGrants(profile),
    onSuccess: () => {
      setShowResults(true);
      success("Grants matched", "AI has found relevant opportunities");
    },
    onError: () => error("Matching failed", "Please try again"),
  });

  const handleMatch = () => {
    if (profile.name && profile.organization) matchMutation.mutate();
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <Target size={20} className="text-neon-mint" /> Grant Matcher
          </h1>
          <p className="text-xs text-slate-400">
            AI‑powered funding opportunity discovery
          </p>
        </div>
        {onClose && (
          <Button size="sm" variant="ghost" onClick={onClose}>
            ✕
          </Button>
        )}
      </div>
      <div className="flex-1 overflow-auto p-4 space-y-4">
        <TerminalCard className="p-4 space-y-4">
          <div className="grid md:grid-cols-2 gap-3">
            <Input
              label="Full Name"
              value={profile.name}
              onChange={(e) => setProfile({ ...profile, name: e.target.value })}
            />
            <Input
              label="Organization / Institution"
              value={profile.organization}
              onChange={(e) =>
                setProfile({ ...profile, organization: e.target.value })
              }
            />
            <Input
              label="Role / Position"
              value={profile.role}
              onChange={(e) => setProfile({ ...profile, role: e.target.value })}
            />
            <Input
              label="Country"
              value={profile.country}
              onChange={(e) =>
                setProfile({ ...profile, country: e.target.value })
              }
            />
          </div>
          <Input
            label="Research Keywords"
            value={profile.research_keywords}
            onChange={(e) =>
              setProfile({ ...profile, research_keywords: e.target.value })
            }
            placeholder="e.g., machine learning, computer vision, NLP"
          />
          <Select
            label="Entity Type"
            options={[
              { value: "academic", label: "Academic / University" },
              { value: "nonprofit", label: "Non‑profit" },
              { value: "startup", label: "Startup" },
              { value: "individual", label: "Individual Researcher" },
            ]}
            value={profile.entity_type}
            onChange={(e) =>
              setProfile({ ...profile, entity_type: e.target.value })
            }
          />
          <SimilaritySlider
            value={profile.min_score}
            onChange={(score) => setProfile({ ...profile, min_score: score })}
          />
          <Button
            onClick={handleMatch}
            disabled={
              !profile.name || !profile.organization || matchMutation.isPending
            }
            className="w-full"
          >
            {matchMutation.isPending ? "Matching..." : "Find Matching Grants"}
          </Button>
        </TerminalCard>
        {matchMutation.isPending && (
          <div className="text-center py-12">
            <Spinner size="lg" />
            <p className="text-slate-400 mt-2">
              AI is matching your profile with funding opportunities...
            </p>
          </div>
        )}
        {showResults && matchMutation.data && (
          <div className="space-y-3">
            <h3 className="text-sm font-bold text-white">Top Matches</h3>
            {matchMutation.data.slice(0, 10).map((grant) => (
              <GrantCard key={grant.id} grant={grant} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
