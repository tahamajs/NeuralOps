export interface ModelLayer {
  name: string;
  type: string;
  params: number;
  color?: string;
  inputShape?: number[];
  outputShape?: number[];
}

export interface ModelArchitecture {
  id: string;
  name: string;
  framework: "onnx" | "pytorch" | "tensorflow";
  layers: ModelLayer[];
  totalParams: number;
}
