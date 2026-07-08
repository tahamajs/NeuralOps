import { request } from "../../../api/client";

export type ModelArchitecture = any;

export const modelViewerAPI = {
  parseModel: (file: File) => {
    const formData = new FormData();
    formData.append("model", file);
    return request.post<ModelArchitecture>("/model-viewer/parse/", formData);
  },
  getModelById: (id: string) =>
    request.get<ModelArchitecture>(`/model-viewer/models/${id}/`),
};
