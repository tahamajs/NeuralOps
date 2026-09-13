const fs = require('fs');
let code = fs.readFileSync('Docs/visualizer/app.js', 'utf8');

const newPipelineFunc = `function getModelPipelineComponents(moduleMeta, modelKey = currentModel) {
    const proj = (moduleMeta && moduleMeta.model_projections && moduleMeta.model_projections[modelKey]) || {};
    const dModel = proj.hidden_size || (modelKey === 'qwen' ? 896 : (modelKey === 'gpt2' ? 768 : 768));
    const dFfn = proj.intermediate_size || (modelKey === 'qwen' ? 4864 : (modelKey === 'gpt2' ? 3072 : 3072));
    const vocabSize = VOCAB_SIZES[modelKey] || 30522;
    
    const isAttn = (moduleMeta && (moduleMeta.category || "").toLowerCase().includes("attention")) || (moduleMeta && (moduleMeta.type || "").toLowerCase().includes("attention"));
    
    let pipeline = [];
    
    pipeline.push({
        id: 'input_embeddings', name: 'INPUT EMBEDDINGS', role: 'Token IDs -> Continuous Vectors', path: 'Stage 1: Token Ingestion',
        shape_transform: \`[\${vocabSize} × \${dModel}]\`, math_latex: 'X^{(0)} = W_{\\\\text{embed}}[T]', color: '#10b981', params: \`\${((vocabSize * dModel) / 1e6).toFixed(1)}M\`, stage: 'Input'
    });

    const groupCustom = (comps, stageName, pathName) => {
        let result = [];
        let qkvGroup = [];
        let gateUpGroup = [];
        
        for (let i = 0; i < comps.length; i++) {
            let c = { ...comps[i], stage: stageName, path: comps[i].path || pathName, math_latex: comps[i].math_latex || '\\\\text{Custom Component}' };
            let id = (c.id || "").toLowerCase();
            
            if ((id.startsWith('q_') || id.startsWith('k_') || id.startsWith('v_')) && !id.includes('factor')) {
                qkvGroup.push(c);
                let nextId = (i < comps.length - 1) ? (comps[i+1].id || "").toLowerCase() : "";
                if (!((nextId.startsWith('q_') || nextId.startsWith('k_') || nextId.startsWith('v_')) && !nextId.includes('factor'))) {
                    if (qkvGroup.length > 1) result.push(qkvGroup);
                    else result.push(qkvGroup[0]);
                    qkvGroup = [];
                }
            } else if (id.startsWith('gate_') || id.startsWith('up_')) {
                gateUpGroup.push(c);
                let nextId = (i < comps.length - 1) ? (comps[i+1].id || "").toLowerCase() : "";
                if (!(nextId.startsWith('gate_') || nextId.startsWith('up_'))) {
                    if (gateUpGroup.length > 1) result.push(gateUpGroup);
                    else result.push(gateUpGroup[0]);
                    gateUpGroup = [];
                }
            } else {
                result.push(c);
            }
        }
        return result;
    };

    if (isAttn && moduleMeta && moduleMeta.components && moduleMeta.components.length > 0) {
        pipeline.push(...groupCustom(moduleMeta.components, 'Attention', 'Path A: Attention Mechanism'));
    } else if (modelKey === 'qwen') {
        pipeline.push([
            { id: 'q_proj', name: 'Q Proj', role: 'Queries', path: 'Path A', shape_transform: \`[\${dModel} × \${dModel}]\`, math_latex: 'Q = X W_q', color: '#38bdf8', params: \`\${((dModel * dModel) / 1000).toFixed(0)}K\`, stage: 'Attention' },
            { id: 'k_proj', name: 'K Proj', role: 'Keys', path: 'Path A', shape_transform: \`[128 × \${dModel}]\`, math_latex: 'K = X W_k', color: '#f59e0b', params: \`\${((128 * dModel) / 1000).toFixed(0)}K\`, stage: 'Attention' },
            { id: 'v_proj', name: 'V Proj', role: 'Values', path: 'Path A', shape_transform: \`[128 × \${dModel}]\`, math_latex: 'V = X W_v', color: '#d97706', params: \`\${((128 * dModel) / 1000).toFixed(0)}K\`, stage: 'Attention' }
        ]);
        pipeline.push({ id: 'o_proj', name: 'O Proj', role: 'Output', path: 'Convergence', shape_transform: \`[\${dModel} × \${dModel}]\`, math_latex: 'Out = \\\\text{Softmax}(Q K^T) V W_o', color: '#2dd4bf', params: \`\${((dModel * dModel) / 1000).toFixed(0)}K\`, stage: 'Attention' });
    } else if (modelKey === 'gpt2') {
        pipeline.push([
            { id: 'c_attn', name: 'c_attn', role: 'Fused QKV', path: 'Path A', shape_transform: \`[3×\${dModel} × \${dModel}]\`, math_latex: '[Q,K,V] = X W_{c\\\\_attn}', color: '#38bdf8', params: \`\${((3*dModel * dModel) / 1000).toFixed(0)}K\`, stage: 'Attention' }
        ]);
        pipeline.push({ id: 'c_proj', name: 'c_proj', role: 'Output', path: 'Convergence', shape_transform: \`[\${dModel} × \${dModel}]\`, math_latex: 'Out = \\\\text{Softmax}(Q K^T) V W_{proj}', color: '#2dd4bf', params: \`\${((dModel * dModel) / 1000).toFixed(0)}K\`, stage: 'Attention' });
    } else {
        pipeline.push([
            { id: 'qkv_proj', name: 'QKV Proj', role: 'QKV Projections', path: 'Path A', shape_transform: \`[3×\${dModel} × \${dModel}]\`, math_latex: '[Q,K,V] = X W', color: '#38bdf8', params: \`\${((3*dModel * dModel) / 1000).toFixed(0)}K\`, stage: 'Attention' }
        ]);
        pipeline.push({ id: 'out_proj', name: 'Output Proj', role: 'Output', path: 'Convergence', shape_transform: \`[\${dModel} × \${dModel}]\`, math_latex: 'Out = \\\\text{Softmax}(Q K^T) V W_o', color: '#2dd4bf', params: \`\${((dModel * dModel) / 1000).toFixed(0)}K\`, stage: 'Attention' });
    }

    pipeline.push({ id: 'attn.residual_add', name: '(+) Residual Add', role: 'Skip Connection', path: 'Highway #1', shape_transform: \`[Seq × \${dModel}]\`, math_latex: 'X_{\\\\text{mid}} = X_{\\\\text{in}} + \\\\text{Attn}(X_{\\\\text{in}})', color: '#0284c7', params: '0 (Op)', stage: 'Residual' });

    if (!isAttn && moduleMeta && moduleMeta.components && moduleMeta.components.length > 0) {
        pipeline.push(...groupCustom(moduleMeta.components, 'FFN/MLP', 'Path B: MLP Mechanism'));
    } else if (modelKey === 'qwen') {
        pipeline.push([
            { id: 'gate_proj', name: 'Gate Proj', role: 'SwiGLU Gate', path: 'Path B', shape_transform: \`[\${dFfn} × \${dModel}]\`, math_latex: 'G = \\\\text{SiLU}(X W_{gate})', color: '#a855f7', params: \`\${((dFfn * dModel) / 1e6).toFixed(1)}M\`, stage: 'MLP' },
            { id: 'up_proj', name: 'Up Proj', role: 'SwiGLU Up', path: 'Path B', shape_transform: \`[\${dFfn} × \${dModel}]\`, math_latex: 'U = X W_{up}', color: '#8b5cf6', params: \`\${((dFfn * dModel) / 1e6).toFixed(1)}M\`, stage: 'MLP' }
        ]);
        pipeline.push({ id: 'down_proj', name: 'Down Proj', role: 'SwiGLU Down', path: 'Convergence', shape_transform: \`[\${dModel} × \${dFfn}]\`, math_latex: 'Y = (G \\\\odot U) W_{down}', color: '#5eead4', params: \`\${((dModel * dFfn) / 1e6).toFixed(1)}M\`, stage: 'MLP' });
    } else {
        pipeline.push([
            { id: 'ffn_up', name: 'FFN Up', role: 'Expansion', path: 'Path B', shape_transform: \`[\${dFfn} × \${dModel}]\`, math_latex: 'H = \\\\text{GELU}(X W_1)', color: '#38bdf8', params: \`\${((dFfn * dModel) / 1e6).toFixed(1)}M\`, stage: 'MLP' }
        ]);
        pipeline.push({ id: 'ffn_down', name: 'FFN Down', role: 'Contraction', path: 'Convergence', shape_transform: \`[\${dModel} × \${dFfn}]\`, math_latex: 'Y = H W_2', color: '#60a5fa', params: \`\${((dModel * dFfn) / 1e6).toFixed(1)}M\`, stage: 'MLP' });
    }

    pipeline.push({ id: 'ffn.residual_add', name: '(+) Residual Add', role: 'Skip Connection', path: 'Highway #2', shape_transform: \`[Seq × \${dModel}]\`, math_latex: 'X_{\\\\text{out}} = X_{\\\\text{mid}} + \\\\text{FFN}(X_{\\\\text{mid}})', color: '#7c3aed', params: '0 (Op)', stage: 'Residual' });
    pipeline.push({ id: 'output_head', name: 'OUTPUT LM HEAD', role: 'Vocabulary Logits', path: 'Stage 7: Output', shape_transform: \`[\${dModel} × \${vocabSize}]\`, math_latex: '\\\\hat{y} = \\\\text{Softmax}(X W_{\\\\text{vocab}}^T)', color: '#f43f5e', params: \`\${((dModel * vocabSize) / 1e6).toFixed(1)}M\`, stage: 'Output' });

    return pipeline;
}`;

let startIdx = code.indexOf('function getModelPipelineComponents(');
let endIdx = code.indexOf('function renderComponentPipeline(');

if (startIdx !== -1 && endIdx !== -1) {
    code = code.substring(0, startIdx) + newPipelineFunc + '\n\n' + code.substring(endIdx);
    fs.writeFileSync('Docs/visualizer/app.js', code);
    console.log("Fixed parallel grouping logic!");
} else {
    console.log("Failed to find bounds");
}
