import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { profileAPI } from "../services/api";
import {
  Card,
  Input,
  Textarea,
  Button,
  Spinner,
  Switch,
} from "../../../components/ui";
import { Save, X } from "lucide-react";
import { ProxySettings } from "../../settings/components/ProxySettings";

export default function ProfileSettingsPage() {
  const queryClient = useQueryClient();
  const { data: profile, isLoading } = useQuery({
    queryKey: ["profile", "me"],
    queryFn: () => profileAPI.getProfile(),
  });

  const [formData, setFormData] = useState({
    full_name: "",
    bio: "",
    location: "",
    website: "",
    company: "",
    github_username: "",
    twitter_username: "",
    linkedin_url: "",
    affiliation: "",
    expertise: [] as string[],
    research_interests: [] as string[],
    pronouns: "",
    timezone: "",
  });

  const [newExpertise, setNewExpertise] = useState("");
  const [newInterest, setNewInterest] = useState("");

  const updateMutation = useMutation({
    mutationFn: (data: any) => profileAPI.updateProfile(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["profile", "me"] });
    },
  });

  if (isLoading) return <Spinner />;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    updateMutation.mutate(formData);
  };

  const addExpertise = () => {
    if (newExpertise && !formData.expertise.includes(newExpertise)) {
      setFormData({
        ...formData,
        expertise: [...formData.expertise, newExpertise],
      });
      setNewExpertise("");
    }
  };

  const removeExpertise = (skill: string) => {
    setFormData({
      ...formData,
      expertise: formData.expertise.filter((s) => s !== skill),
    });
  };

  // ... similar for research interests

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      <h1 className="text-2xl font-bold text-white">Profile Settings</h1>
      <ProxySettings />

      <Card>
        <form onSubmit={handleSubmit} className="space-y-6">
          <Input
            label="Full Name"
            value={formData.full_name}
            onChange={(e) =>
              setFormData({ ...formData, full_name: e.target.value })
            }
          />
          <Textarea
            label="Bio"
            rows={4}
            value={formData.bio}
            onChange={(e) => setFormData({ ...formData, bio: e.target.value })}
          />
          <Input
            label="Location"
            value={formData.location}
            onChange={(e) =>
              setFormData({ ...formData, location: e.target.value })
            }
          />
          <Input
            label="Website"
            value={formData.website}
            onChange={(e) =>
              setFormData({ ...formData, website: e.target.value })
            }
          />
          <Input
            label="Company"
            value={formData.company}
            onChange={(e) =>
              setFormData({ ...formData, company: e.target.value })
            }
          />
          <div className="space-y-2">
            <label className="text-xs font-bold text-slate-400">
              Expertise
            </label>
            <div className="flex gap-2">
              <Input
                value={newExpertise}
                onChange={(e) => setNewExpertise(e.target.value)}
                placeholder="e.g., Computer Vision"
              />
              <Button type="button" size="sm" onClick={addExpertise}>
                Add
              </Button>
            </div>
            <div className="flex flex-wrap gap-2 mt-2">
              {formData.expertise.map((skill) => (
                <span
                  key={skill}
                  className="px-2 py-1 rounded bg-neon-blue/10 text-neon-blue text-xs flex items-center gap-1"
                >
                  {skill}{" "}
                  <button type="button" onClick={() => removeExpertise(skill)}>
                    <X size={12} />
                  </button>
                </span>
              ))}
            </div>
          </div>
          <Button type="submit" disabled={updateMutation.isPending}>
            Save Changes
          </Button>
        </form>
      </Card>
    </div>
  );
}
