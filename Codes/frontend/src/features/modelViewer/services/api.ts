import api from "../../../api/client";

export const modelViewerAPI = {
  parseModel: (file: File) => {
    const formData = new FormData();
    formData.append("model", file);
    return api.post<ModelArchitecture>("/model-viewer/parse/", formData);
  },
  getModelById: (id: string) =>
    api.get<ModelArchitecture>(`/model-viewer/models/${id}/`),
};
