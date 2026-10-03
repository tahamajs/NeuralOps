/**
 * educational/model_matrix_dashboard.js
 * =====================================
 * Comprehensive 6-Model Architecture & Benchmark Comparison Matrix Modal:
 * - DistilBERT, RoBERTa, GPT-2, Qwen 2.5 (GQA), ToMoE (MoE), SliceGPT (Pruned)
 * - Complete breakdown of parameters, hidden dimensions, heads, KV cache footprint, and SST-2 accuracy
 * - Exportable CSV functionality
 */

export const MATRIX_MODELS = [
    {
        key: 'distilbert',
        name: 'DistilBERT Base',
        family: 'Encoder',
        params: '66.36M',
        hidden_dim: 768,
        layers: 6,
        heads: '12 (MHA)',
        kv_heads: '12',
        kv_cache_per_tok: '1,536 B',
        accuracy_sst2: '92.4%',
        vram_mb: '280 MB',
        tokens_sec: '1,420 tok/s',
        paradigm: 'Dense vs SVD'
    },
    {
        key: 'roberta',
        name: 'RoBERTa Base',
        family: 'Encoder',
        params: '124.65M',
        hidden_dim: 768,
        layers: 12,
        heads: '12 (MHA)',
        kv_heads: '12',
        kv_cache_per_tok: '1,536 B',
        accuracy_sst2: '94.8%',
        vram_mb: '520 MB',
        tokens_sec: '890 tok/s',
        paradigm: 'Dense vs SVD'
    },
    {
        key: 'gpt2',
        name: 'GPT-2 Small',
        family: 'Causal Decoder',
        params: '124.44M',
        hidden_dim: 768,
        layers: 12,
        heads: '12 (Fused QKV)',
        kv_heads: '12',
        kv_cache_per_tok: '1,536 B',
        accuracy_sst2: '90.2%',
        vram_mb: '510 MB',
        tokens_sec: '950 tok/s',
        paradigm: 'Fused c_attn'
    },
    {
        key: 'qwen',
        name: 'Qwen 2.5 (0.5B)',
        family: 'Causal Decoder',
        params: '490.00M',
        hidden_dim: 896,
        layers: 24,
        heads: '14 (GQA)',
        kv_heads: '2 (Grouped)',
        kv_cache_per_tok: '256 B (-83.3%)',
        accuracy_sst2: '93.5%',
        vram_mb: '1,250 MB',
        tokens_sec: '780 tok/s',
        paradigm: 'Grouped-Query'
    },
    {
        key: 'tomoe',
        name: 'ToMoE Experts',
        family: 'Sparse MoE',
        params: '110.50M',
        hidden_dim: 768,
        layers: 6,
        heads: '12 (MHA)',
        kv_heads: '12',
        kv_cache_per_tok: '1,536 B',
        accuracy_sst2: '91.8%',
        vram_mb: '440 MB',
        tokens_sec: '1,120 tok/s',
        paradigm: '4-Way Top-2 MoE'
    },
    {
        key: 'slicegpt',
        name: 'SliceGPT Pruned',
        family: 'Dense Sliced',
        params: '49.80M',
        hidden_dim: 576,
        layers: 6,
        heads: '12 (Sliced)',
        kv_heads: '12',
        kv_cache_per_tok: '1,152 B (-25%)',
        accuracy_sst2: '89.6%',
        vram_mb: '210 MB',
        tokens_sec: '1,890 tok/s',
        paradigm: '25% Channel Pruned'
    }
];

export function openModelMatrixDashboard(onSelectModel) {
    let modal = document.getElementById('model-matrix-dashboard-modal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'model-matrix-dashboard-modal';
        modal.className = 'modal-overlay hidden';
        modal.setAttribute('role', 'dialog');
        modal.setAttribute('aria-modal', 'true');
        modal.innerHTML = `
            <div class="modal-container" style="max-width: 1080px; max-height: 88vh; display:flex; flex-direction:column;">
                <div class="modal-header">
                    <div>
                        <span class="hud-badge" style="background:#0284c7; color:#fff;">CROSS-ARCHITECTURE BENCHMARK MATRIX</span>
                        <h2>6-Model Architectural &amp; Hardware Telemetry</h2>
                        <p class="modal-subtitle">Side-by-side comparison across parameter counts, head configurations, KV cache footprint, and SST-2 accuracy.</p>
                    </div>
                    <div style="display:flex; align-items:center; gap:10px;">
                        <button id="btn-export-matrix-csv" class="round-button" style="background:rgba(16,185,129,0.2); color:#34d399; font-size:11px;">📥 Export CSV</button>
                        <button id="btn-close-matrix-modal" class="btn-icon-only modal-close">✕</button>
                    </div>
                </div>
                <div style="padding:16px 24px; overflow-x:auto; flex:1;">
                    <table class="mmd-table">
                        <thead>
                            <tr>
                                <th class="mmd-th">Model</th>
                                <th class="mmd-th">Family</th>
                                <th class="mmd-th">Parameters</th>
                                <th class="mmd-th">Hidden Dim</th>
                                <th class="mmd-th">Attention Heads</th>
                                <th class="mmd-th">KV Heads</th>
                                <th class="mmd-th">KV Cache / Tok</th>
                                <th class="mmd-th">SST-2 Accuracy</th>
                                <th class="mmd-th">Latency</th>
                                <th class="mmd-th">Action</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${MATRIX_MODELS.map(m => `
                                <tr class="mmd-row">
                                    <td class="mmd-td" style="font-weight:700; color:#fff;">${m.name}</td>
                                    <td class="mmd-td" style="color:var(--text-muted);">${m.family}</td>
                                    <td class="mmd-td" style="color:#38bdf8; font-weight:600;">${m.params}</td>
                                    <td class="mmd-td">${m.hidden_dim}d</td>
                                    <td class="mmd-td">${m.heads}</td>
                                    <td class="mmd-td" style="color:#f59e0b; font-weight:600;">${m.kv_heads}</td>
                                    <td class="mmd-td" style="color:#34d399;">${m.kv_cache_per_tok}</td>
                                    <td class="mmd-td" style="color:#5eead4; font-weight:700;">${m.accuracy_sst2}</td>
                                    <td class="mmd-td" style="color:var(--text-faint);">${m.tokens_sec}</td>
                                    <td class="mmd-td">
                                        <button class="round-button btn-load-model" data-model="${m.key}" style="font-size:10px; padding:2px 8px;">Load 3D</button>
                                    </td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            </div>
        `;
        document.body.appendChild(modal);

        modal.querySelector('#btn-close-matrix-modal').addEventListener('click', () => modal.classList.add('hidden'));
        modal.addEventListener('click', (e) => {
            if (e.target === modal) modal.classList.add('hidden');
        });

        modal.querySelector('#btn-export-matrix-csv').addEventListener('click', () => {
            const headers = ['Model', 'Family', 'Parameters', 'Hidden_Dim', 'Layers', 'Heads', 'KV_Heads', 'KV_Cache_Per_Token', 'SST2_Accuracy', 'Latency', 'Paradigm'];
            const rows = MATRIX_MODELS.map(m => [
                `"${m.name}"`,
                `"${m.family}"`,
                `"${m.params}"`,
                m.hidden_dim,
                m.layers,
                `"${m.heads}"`,
                `"${m.kv_heads}"`,
                `"${m.kv_cache_per_tok}"`,
                `"${m.accuracy_sst2}"`,
                `"${m.tokens_sec}"`,
                `"${m.paradigm}"`
            ]);
            const csv = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
            const blob = new Blob([csv], { type: 'text/csv' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'transformer_architecture_comparison_matrix.csv';
            a.click();
            URL.revokeObjectURL(url);
        });

        modal.querySelectorAll('.btn-load-model').forEach(b => {
            b.addEventListener('click', () => {
                modal.classList.add('hidden');
                if (onSelectModel) onSelectModel(b.dataset.model);
            });
        });
    }

    modal.classList.remove('hidden');
}
