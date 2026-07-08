import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import api from "../../../api/client";
import { Card, Input, Button, Spinner, Avatar } from "../../../components/ui";
import { useAuthStore } from "../../auth/store/authStore";
import { useState } from "react";

export default function UserProfilePage() {
  const { user } = useAuthStore();
  const queryClient = useQueryClient();
  const [editMode, setEditMode] = useState(false);
  const [bio, setBio] = useState("");
  const [avatarFile, setAvatarFile] = useState<File | null>(null);

  const { data: profile, isLoading } = useQuery({
    queryKey: ["profile", user?.username],
    queryFn: async () => (await api.get("/profiles/me/")).data,
    enabled: !!user,
  });

  const updateProfile = useMutation({
    mutationFn: async (data: any) => api.patch("/profiles/me/", data),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ["profile", user?.username] }),
  });

  const uploadAvatar = useMutation({
    mutationFn: async (file: File) => {
      const form = new FormData();
      form.append("avatar", file);
      return api.post("/profiles/me/avatar/", form);
    },
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ["profile", user?.username] }),
  });

  if (isLoading) return <Spinner />;

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      <Card className="text-center">
        <Avatar src={profile?.avatar_url} size="lg" className="mx-auto mb-3" />
        <h1 className="text-2xl font-bold text-white">{profile?.username}</h1>
        <p className="text-slate-400">{profile?.email}</p>
        <Button
          size="sm"
          variant="ghost"
          className="mt-3"
          onClick={() => setEditMode(!editMode)}
        >
          Edit Profile
        </Button>
      </Card>

      {editMode ? (
        <Card>
          <Input
            label="Bio"
            value={bio}
            onChange={(e) => setBio(e.target.value)}
            placeholder="Tell us about yourself"
          />
          <input
            type="file"
            accept="image/*"
            onChange={(e) => setAvatarFile(e.target.files?.[0] || null)}
          />
          <Button onClick={() => updateProfile.mutate({ bio })}>Save</Button>
          <Button
            variant="ghost"
            onClick={() => uploadAvatar.mutate(avatarFile!)}
            disabled={!avatarFile}
          >
            Upload Avatar
          </Button>
        </Card>
      ) : (
        <Card>
          <h2 className="text-sm font-bold text-white">Bio</h2>
          <p className="text-slate-300">{profile?.bio || "No bio yet."}</p>
        </Card>
      )}
    </div>
  );
}
