import { useState, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import api from "../../../api/client";
import { Card, Input, Button, Modal } from "../../../components/ui";

export function OrganizationSettings({ orgId }: { orgId: string }) {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [deleteConfirm, setDeleteConfirm] = useState("");
  const [name, setName] = useState("");
  const [slug, setSlug] = useState("");
  const [description, setDescription] = useState("");
  const [avatarFile, setAvatarFile] = useState<File | null>(null);

  const { data: org } = useQuery({
    queryKey: ["organization", orgId],
    queryFn: async () => (await api.get(`/orgs/${orgId}/`)).data as any,
  });

  useEffect(() => {
    if (org) {
      setName(org.name);
      setSlug(org.slug);
      setDescription(org.description || "");
    }
  }, [org]);

  const updateOrg = useMutation({
    mutationFn: async () => {
      const form = new FormData();
      form.append("name", name);
      form.append("slug", slug);
      form.append("description", description);
      if (avatarFile) form.append("avatar", avatarFile);
      return api.patch(`/orgs/${orgId}/`, form);
    },
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ["organization", orgId] }),
  });

  const deleteOrg = useMutation({
    mutationFn: () => api.delete(`/orgs/${orgId}/`),
    onSuccess: () => navigate("/organizations"),
  });

  return (
    <div className="space-y-6 max-w-2xl">
      <Card>
        <h3 className="text-sm font-bold text-white mb-3">Profile Settings</h3>
        <div className="space-y-4">
          <Input
            label="Organization Name"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
          <Input
            label="Slug"
            value={slug}
            onChange={(e) => setSlug(e.target.value)}
          />
          <Input
            label="Description"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
          />
          <div>
            <label className="text-xs font-bold text-slate-400">Avatar</label>
            <input
              type="file"
              accept="image/*"
              onChange={(e) => setAvatarFile(e.target.files?.[0] || null)}
            />
          </div>
          <Button onClick={() => updateOrg.mutate()}>Save Changes</Button>
        </div>
      </Card>

      <Card className="border-red-500/30">
        <h3 className="text-sm font-bold text-red-400 mb-3">Danger Zone</h3>
        <p className="text-sm text-slate-400 mb-3">
          Once you delete this organization, all data will be permanently lost.
        </p>
        <Button variant="danger" onClick={() => setShowDeleteModal(true)}>
          Delete Organization
        </Button>
      </Card>

      <Modal
        open={showDeleteModal}
        onClose={() => setShowDeleteModal(false)}
        title="Delete Organization"
      >
        <p>
          Type <strong>{org?.name}</strong> to confirm:
        </p>
        <Input
          value={deleteConfirm}
          onChange={(e) => setDeleteConfirm(e.target.value)}
          className="mt-2"
        />
        <div className="flex gap-2 mt-4">
          <Button variant="ghost" onClick={() => setShowDeleteModal(false)}>
            Cancel
          </Button>
          <Button
            variant="danger"
            disabled={deleteConfirm !== org?.name}
            onClick={() => deleteOrg.mutate()}
          >
            Delete Permanently
          </Button>
        </div>
      </Modal>
    </div>
  );
}
