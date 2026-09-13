import re

with open('Docs/visualizer/app.js', 'r') as f:
    content = f.read()

# Replace getModelPipelineComponents
new_func = """function getModelPipelineComponents(moduleMeta, modelKey = currentModel) {
    const proj = (moduleMeta && moduleMeta.model_projections && moduleMeta.model_projections[modelKey]) || {};
    const dModel = proj.hidden_size || (modelKey === 'qwen' ? 896 : 768);
    const dFfn = proj.intermediate_size || (modelKey === 'qwen' ? 4864 : 3072);
    const vocabSize = VOCAB_SIZES[modelKey] || 30522;
    const isAttn = (moduleMeta && (moduleMeta.category || "").toLowerCase().includes("attention")) || (moduleMeta && (moduleMeta.type || "").toLowerCase().includes("attention"));
    
    // Build the core sequence
    let pipeline = [];
    
    // 1. INPUT EMBEDDINGS
    pipeline.push({
        id: 'input_embeddings',
        name: 'INPUT EMBEDDINGS',
        role: 'Token IDs -> Continuous Vectors',
        path: 'Stage 1: Token Ingestion',
        shape_transform: `[${vocabSize} × ${dModel}]`,
        math_latex: 'X^{(0)} = W_{\\text{embed}}[T]',
        color: '#10b981',
        params: `${((vocabSize * dModel) / 1e6).toFixed(1)}M`,
        stage: 'Input'
    });

    // 2. ATTENTION OR COMPRESSED ATTENTION
    if (isAttn && moduleMeta && moduleMeta.components && moduleMeta.components.length > 0) {
        moduleMeta.components.forEach(c => {
            pipeline.push({
                ...c,
                stage: 'Attention',
                path: c.path || 'Path A: Attention Mechanism',
                math_latex: c.math_latex || '\\text{Custom Attention Component}'
            });
        });
    } else if (modelKey === 'qwen') {
        pipeline.push({ id: 'q_proj', name: 'Q Proj', role: 'Queries', path: 'Path A', shape_transform: `[${dModel} × ${dModel}]`, color: '#38bdf8', params: `${((dModel * dModel) / 1000).toFixed(0)}K`, stage: 'Attention' });
        pipeline.push({ id: 'k_proj', name: 'K Proj', role: 'Keys', path: 'Path A', shape_transform: `[128 × ${dModel}]`, color: '#f59e0b', params: `${((128 * dModel) / 1000).toFixed(0)}K`, stage: 'Attention' });
        pipeline.push({ id: 'v_proj', name: 'V Proj', role: 'Values', path: 'Path A', shape_transform: `[128 × ${dModel}]`, color: '#d97706', params: `${((128 * dModel) / 1000).toFixed(0)}K`, stage: 'Attention' });
        pipeline.push({ id: 'o_proj', name: 'O Proj', role: 'Output', path: 'Convergence', shape_transform: `[${dModel} × ${dModel}]`, color: '#2dd4bf', params: `${((dModel * dModel) / 1000).toFixed(0)}K`, stage: 'Attention' });
    } else {
        pipeline.push({ id: 'qkv_proj', name: 'QKV Proj', role: 'QKV Projections', path: 'Path A', shape_transform: `[3×${dModel} × ${dModel}]`, color: '#38bdf8', params: `${((3*dModel * dModel) / 1000).toFixed(0)}K`, stage: 'Attention' });
        pipeline.push({ id: 'out_proj', name: 'Output Proj', role: 'Output', path: 'Convergence', shape_transform: `[${dModel} × ${dModel}]`, color: '#2dd4bf', params: `${((dModel * dModel) / 1000).toFixed(0)}K`, stage: 'Attention' });
    }

    pipeline.push({
        id: 'attn.residual_add',
        name: '(+) Residual Add',
        role: 'Skip Connection',
        path: 'Highway #1',
        shape_transform: `[Seq × ${dModel}]`,
        math_latex: 'X_{\\text{mid}} = X_{\\text{in}} + \\text{Attn}(X_{\\text{in}})',
        color: '#0284c7',
        params: '0 (Op)',
        stage: 'Residual'
    });

    // 3. FFN OR COMPRESSED FFN
    if (!isAttn && moduleMeta && moduleMeta.components && moduleMeta.components.length > 0) {
        moduleMeta.components.forEach(c => {
            pipeline.push({
                ...c,
                stage: 'FFN/MLP',
                path: c.path || 'Path B: MLP Mechanism',
                math_latex: c.math_latex || '\\text{Custom FFN Component}'
            });
        });
    } else if (modelKey === 'qwen') {
        pipeline.push({ id: 'gate_proj', name: 'Gate Proj', role: 'SwiGLU Gate', path: 'Path B', shape_transform: `[${dFfn} × ${dModel}]`, color: '#a855f7', params: `${((dFfn * dModel) / 1e6).toFixed(1)}M`, stage: 'MLP' });
        pipeline.push({ id: 'up_proj', name: 'Up Proj', role: 'SwiGLU Up', path: 'Path B', shape_transform: `[${dFfn} × ${dModel}]`, color: '#8b5cf6', params: `${((dFfn * dModel) / 1e6).toFixed(1)}M`, stage: 'MLP' });
        pipeline.push({ id: 'down_proj', name: 'Down Proj', role: 'SwiGLU Down', path: 'Convergence', shape_transform: `[${dModel} × ${dFfn}]`, color: '#5eead4', params: `${((dModel * dFfn) / 1e6).toFixed(1)}M`, stage: 'MLP' });
    } else {
        pipeline.push({ id: 'ffn_up', name: 'FFN Up', role: 'Expansion', path: 'Path B', shape_transform: `[${dFfn} × ${dModel}]`, color: '#38bdf8', params: `${((dFfn * dModel) / 1e6).toFixed(1)}M`, stage: 'MLP' });
        pipeline.push({ id: 'ffn_down', name: 'FFN Down', role: 'Contraction', path: 'Convergence', shape_transform: `[${dModel} × ${dFfn}]`, color: '#60a5fa', params: `${((dModel * dFfn) / 1e6).toFixed(1)}M`, stage: 'MLP' });
    }

    pipeline.push({
        id: 'ffn.residual_add',
        name: '(+) Residual Add',
        role: 'Skip Connection',
        path: 'Highway #2',
        shape_transform: `[Seq × ${dModel}]`,
        math_latex: 'X_{\\text{out}} = X_{\\text{mid}} + \\text{FFN}(X_{\\text{mid}})',
        color: '#7c3aed',
        params: '0 (Op)',
        stage: 'Residual'
    });

    pipeline.push({
        id: 'output_head',
        name: 'OUTPUT LM HEAD',
        role: 'Vocabulary Logits',
        path: 'Stage 7: Output',
        shape_transform: `[${dModel} × ${vocabSize}]`,
        math_latex: '\\hat{y} = \\text{Softmax}(X W_{\\text{vocab}}^T)',
        color: '#f43f5e',
        params: `${((dModel * vocabSize) / 1e6).toFixed(1)}M`,
        stage: 'Output'
    });

    return pipeline;
}"""

# Do regex replacement
import re
pattern = re.compile(r'function getModelPipelineComponents.*?^}', re.MULTILINE | re.DOTALL)
content = pattern.sub(new_func, content, count=1)

with open('Docs/visualizer/app.js', 'w') as f:
    f.write(content)
