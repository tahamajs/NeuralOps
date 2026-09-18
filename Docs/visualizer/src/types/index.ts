/**
 * types/index.ts
 * ==============
 * Comprehensive TypeScript interfaces and contracts for the 3D Neural Architecture Visualizer.
 * Covers 3D nodes, tensor geometries, module metadata, model specifications, token physics, and benchmarks.
 */

export type ModelKey = 'distilbert' | 'roberta' | 'gpt2' | 'qwen';
export type ParadigmKey = 'svd' | 'moe' | 'mla' | 'slicegpt' | 'quant' | 'ssm';

// ── Node & Scene Graph Types ───────────────────────────────────────

export interface Node3D {
    id: string;
    type: 'dense' | 'svd' | 'moe' | 'mla' | 'embedding' | 'output_head' | 'residual_add';
    sub_type?: string;
    parent_layer_index: number;
    parent_layer_name: string;
    sequence_index: number;
    x: number;
    y: number;
    z: number;
    width: number;
    height: number;
    depth: number;
    label: string;
    metadata: NodeMetadata;
    isMacro?: boolean;
    custom_color?: string;
}

export interface NodeMetadata {
    shape?: [number, number] | [number] | string[];
    params?: string;
    params_raw?: number;
    storage?: string;
    storage_bytes?: number;
    dense_equivalent_params?: number;
    dense_equivalent_params_formatted?: string;
    compression_ratio?: number;
    compression?: string;
    type?: string;
    rank?: number;
    heads?: number;
    num_experts?: number;
    expert_idx?: number;
    role?: string;
}

export interface MacroNode {
    id: string;
    name: string;
    parent_layer_idx: number;
    parent_layer_name: string;
    width: number;
    height: number;
    depth: number;
    x: number;
    y: number;
    z: number;
    metadata: {
        shape: [number];
        params: string;
        storage: string;
    };
}

export interface Edge3D {
    from_id: string;
    to_id: string;
    curve_points?: [number, number, number][];
    type?: 'dataflow' | 'residual_bypass';
}

// ── Module & Component Types ───────────────────────────────────────

export interface ModuleMeta {
    id: string;
    name: string;
    file: string;
    type: string;
    category: string;
    paradigm: ParadigmKey;
    icon: string;
    description: string;
    classes?: string[];
    methods_count?: number;
    math?: string;
    math_latex?: string;
    markdown_preview?: string;
    components?: SublayerComponent[];
    model_projections?: Record<ModelKey, ModelProjection>;
    verified_benchmark?: BenchmarkResults;
}

export interface SublayerComponent {
    id: string;
    name: string;
    role: string;
    shape_transform: string;
    color: string;
    path?: string;
    stage?: string;
    params?: string;
}

export interface ModelProjection {
    model_name: string;
    family: string;
    hidden_size: number;
    intermediate_size: number;
    num_layers: number;
    baseline_layer_params: number;
    compressed_layer_p: number;
    total_baseline_params: number;
    total_compressed_params: number;
    savings_ratio: number;
    storage_saved_bytes: number;
    storage_saved_formatted: string;
    benchmark?: BenchmarkResults;
}

export interface BenchmarkResults {
    status: 'PASS' | 'PASS_LOW' | 'FAIL' | 'SKIP' | 'PENDING';
    accuracy_pct?: number;
    f1_score?: number;
    peak_vram_mb?: number;
    throughput_tokens_sec?: number;
    source_file?: string;
}

// ── Token Physics & Trajectory Types ───────────────────────────────

export interface TokenState {
    y: number;
    speed: number;
    baseX: number;
    lateralOffset: number;
    lateralNoise: number;
    isResidual: boolean;
    subStream: 'q' | 'k' | 'v';
    swigluStream: 'gate' | 'up';
    headIdx: number;
    expertTarget: number;
    pulsePhase: number;
}

export interface TokenPosition {
    x: number;
    z: number;
    sublayer: string;
}

// ── Spatial Data Container Types ───────────────────────────────────

export interface SpatialData {
    title: string;
    generated_at: string;
    models: Record<ModelKey, ModelSpatialData>;
}

export interface ModelSpatialData {
    true_num_layers: number;
    capped_num_layers: number;
    baseline_nodes: Node3D[];
    baseline_macro_nodes?: MacroNode[];
    compressed_nodes: Node3D[];
    macro_nodes?: MacroNode[];
    edges: Edge3D[];
    max_y: number;
    num_baseline: number;
    num_compressed: number;
    total_baseline_params: number;
    total_baseline_params_formatted: string;
    total_compressed_params: number;
    total_compressed_params_formatted: string;
    overall_compression_ratio: number;
    storage_saved_bytes: number;
    storage_saved_formatted: string;
    modules: Record<string, ModuleData>;
}

export interface ModuleData {
    compressed_nodes: Node3D[];
    macro_nodes?: MacroNode[];
    name: string;
    total_compressed_params_formatted?: string;
    overall_compression_ratio?: number;
    storage_saved_formatted?: string;
}

// ── Application Observable State ───────────────────────────────────

export interface AppState {
    currentModel: ModelKey;
    currentCompression: string;
    activeResearchModule: ModuleMeta | null;
    pinnedModule: ModuleMeta | null;
    simulationSpeed: number;
    isPlaying: boolean;
    enablePulses: boolean;
    activeNodeId: string | null;
    lodLevel: 'macro' | 'mid' | 'micro';
    isTourActive: boolean;
    tourStep: number;
}
