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

    if (isAttn && moduleMeta && moduleMeta.components && moduleMeta.components.length > 0) {
        pipeline.push(moduleMeta.components.map(c => ({
            ...c, stage: 'Attention', path: c.path || 'Path A: Attention Mechanism', math_latex: c.math_latex || '\\\\text{Custom Attention}'
        })));
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
        pipeline.push(moduleMeta.components.map(c => ({
            ...c, stage: 'FFN/MLP', path: c.path || 'Path B: MLP Mechanism', math_latex: c.math_latex || '\\\\text{Custom FFN Component}'
        })));
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

const newRenderFunc = `function renderComponentPipeline(moduleMeta, modelKey = currentModel) {
    const listEl = document.getElementById('component-pipeline-list');
    const badgeEl = document.getElementById('component-count-badge');
    if (!listEl) return;

    const comps = getModelPipelineComponents(moduleMeta, modelKey);
    let totalBlocks = 0;
    
    // Flatten for counting and highlighting logic
    const flatComps = [];
    comps.forEach(c => {
        if (Array.isArray(c)) { totalBlocks += c.length; flatComps.push(...c); }
        else { totalBlocks += 1; flatComps.push(c); }
    });

    if (badgeEl) badgeEl.innerText = \`\${totalBlocks} BLOCKS\`;

    if (totalBlocks === 0) {
        listEl.innerHTML = \`<div style="font-size: 10px; color: var(--text-muted); padding: 6px;">Single Unified Linear Projection</div>\`;
        return;
    }

    const renderItem = (c) => \`
        <div class="component-pipe-item \${Array.isArray(c) ? 'nested-item' : ''}" data-comp-id="\${c.id}" onclick="selectPartIn3D('\${c.id}')" title="Click to illuminate \${c.name} in 3D & view mathematical specs" style="--cpi-color: \${c.color || '#2dd4bf'};">
            <div class="cpi-left">
                <span class="cpi-swatch" style="background: \${c.color || '#2dd4bf'};"></span>
                <div class="cpi-info">
                    <div class="cpi-name-row">
                        <span class="cpi-name">\${c.name}</span>
                        <span class="cpi-path-tag">\${c.path}</span>
                    </div>
                    <span class="cpi-role">\${c.role}</span>
                    <span class="cpi-math-preview" data-raw-math="\${encodeURIComponent(c.math_latex || '')}">$$ \${c.math_latex || ''} $$</span>
                </div>
            </div>
            <div class="cpi-right">
                <span class="cpi-shape">\${c.shape_transform || ''}</span>
                <span class="cpi-params">\${c.params || ''}</span>
            </div>
        </div>
    \`;

    listEl.innerHTML = comps.map(group => {
        if (Array.isArray(group)) {
            return \`<div class="pipeline-parallel-group">
                <div class="parallel-indicator"></div>
                \${group.map(c => renderItem(c)).join('')}
            </div>\`;
        } else {
            return renderItem(group);
        }
    }).join('');

    if (window.katex) {
        listEl.querySelectorAll('.cpi-math-preview').forEach(el => {
            const raw = decodeURIComponent(el.getAttribute('data-raw-math') || '');
            if (raw) {
                try { katex.render(raw, el, { displayMode: false, throwOnError: false }); } catch(e) {}
            }
        });
    }

    if (activeSelectedSublayerKey) {
        syncUiPartHighlights(activeSelectedSublayerKey);
    }
}`;

let startIdx = code.indexOf('function getModelPipelineComponents(');
let endIdx = code.indexOf('function renderTokenLifecycle(');

if (startIdx !== -1 && endIdx !== -1) {
    let before = code.substring(0, startIdx);
    let after = code.substring(endIdx);
    
    // We also need to replace renderComponentPipeline which is below renderTokenLifecycle
    let codePart2 = after;
    let renderStart = codePart2.indexOf('function renderComponentPipeline(');
    let renderEnd = codePart2.indexOf('window.focusComponentIn3D = ');
    
    if (renderStart !== -1 && renderEnd !== -1) {
        let finalBefore = before + newPipelineFunc + '\n\n' + codePart2.substring(0, renderStart);
        let finalAfter = '\n\n' + codePart2.substring(renderEnd);
        fs.writeFileSync('Docs/visualizer/app.js', finalBefore + newRenderFunc + finalAfter);
        console.log("Patched multi-path pipeline logic!");
    } else {
        console.log("Failed to find renderComponentPipeline bounds");
    }
} else {
    console.log("Failed to find getModelPipelineComponents bounds");
}
