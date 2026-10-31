/**
 * interactive/token_prediction.js
 * ===============================
 * Live Autoregressive Token Prediction Playground.
 * Predicts next-token probability distributions from partial sentences,
 * with temperature scaling, top-k filtering, greedy vs sampling modes,
 * 3D/SVG probability bars, and click-to-append generation.
 */

// Canonical vocabulary subset for deterministic in-browser prediction
export const VOCAB_SUBSET = [
    'transformer', 'model', 'attention', 'neural', 'network', 'learning',
    'deep', 'language', 'tokens', 'layers', 'weights', 'data', 'representation',
    'vector', 'matrix', 'compression', 'heads', 'inference', 'training', 'accuracy',
    'is', 'the', 'a', 'in', 'of', 'to', 'for', 'and', 'with', 'on', 'by',
    'efficient', 'fast', 'scalable', 'compact', 'sparse', 'dense', 'low-rank',
    'generates', 'predicts', 'computes', 'optimizes', 'encodes', 'decodes'
];

/**
 * Computes next-token probability distribution with temperature and top-k filtering.
 * @param {string} prompt - Partial input sentence
 * @param {number} temperature - Softmax temperature T (0.1 to 2.0)
 * @param {number} topK - Top-k candidate truncation (1 to 50)
 * @returns {Array<object>} Top-10 sorted candidate tokens with logits and probabilities
 */
export function predictNextToken(prompt = '', temperature = 1.0, topK = 10) {
    const words = (prompt || '').trim().toLowerCase().split(/\s+/).filter(Boolean);
    const lastWord = words[words.length - 1] || '';
    const temp = Math.max(0.1, Math.min(3.0, temperature || 1.0));
    const kLimit = Math.max(1, Math.min(VOCAB_SUBSET.length, topK || 10));

    // Compute semantic logits based on vocabulary character & bi-gram affinity
    const rawScores = VOCAB_SUBSET.map((candidate, idx) => {
        let score = 0;
        // Bigram affinity heuristics with distinct logit separation
        if (lastWord === 'the') {
            if (candidate === 'model') score += 5.8;
            else if (['transformer', 'network', 'tokens'].includes(candidate)) score += 3.0;
        } else if (lastWord === 'model') {
            if (candidate === 'parameters') score += 6.5;
            else if (['weights', 'architecture', 'accuracy', 'layers'].includes(candidate)) score += 2.5;
        } else if (lastWord === 'attention') {
            if (candidate === 'heads') score += 6.0;
            else if (['weights', 'matrix', 'tokens'].includes(candidate)) score += 3.0;
        } else if (lastWord === 'neural') {
            if (candidate === 'network') score += 6.5;
            else if (['model', 'representation'].includes(candidate)) score += 2.8;
        } else if (lastWord === 'deep') {
            if (candidate === 'learning') score += 6.5;
            else if (['network', 'neural', 'layers'].includes(candidate)) score += 2.8;
        } else if (lastWord === 'is') {
            if (candidate === 'a') score += 5.5;
            else if (['the', 'efficient', 'fast'].includes(candidate)) score += 2.8;
        } else if (['dense', 'low-rank', 'sparse'].includes(lastWord)) {
            if (candidate === 'matrix') score += 6.0;
            else if (['compression', 'layers', 'weights'].includes(candidate)) score += 2.5;
        }

        // Phonetic/character smooth variance
        const charSeed = candidate.charCodeAt(0) + (lastWord.charCodeAt(0) || 100);
        score += Math.sin(charSeed * 0.4 + idx * 0.3) * 1.5 + 2.0;

        return {
            token: candidate,
            logit: score
        };
    });

    // Sort by logit descending
    rawScores.sort((a, b) => b.logit - a.logit);

    // Apply Top-K Truncation
    const candidates = rawScores.slice(0, kLimit);

    // Apply Temperature Scaled Softmax
    const maxLogit = Math.max(...candidates.map(c => c.logit / temp));
    let sumExp = 0;
    candidates.forEach(c => {
        c.exp = Math.exp(c.logit / temp - maxLogit);
        sumExp += c.exp;
    });

    const invSum = 1.0 / (sumExp || 1.0);
    candidates.forEach(c => {
        c.probability = c.exp * invSum;
    });

    return candidates.slice(0, 10);
}

/**
 * Initializes and displays the Live Token Prediction Playground Modal.
 * @param {Function} onTokenSelected - Callback when a token is chosen
 */
export function openTokenPredictionPlayground(onTokenSelected) {
    let modal = document.getElementById('token-prediction-modal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'token-prediction-modal';
        modal.className = 'modal-overlay hidden';
        modal.setAttribute('role', 'dialog');
        modal.setAttribute('aria-modal', 'true');
        modal.innerHTML = `
            <div class="modal-container" style="max-width: 920px; max-height: 88vh; display:flex; flex-direction:column;">
                <div class="modal-header">
                    <div>
                        <span class="hud-badge" style="background:#10b981; color:#fff;">FEATURE 1.2 · LIVE TOKEN PREDICTION</span>
                        <h2>Autoregressive Generation &amp; Logit Distribution</h2>
                        <p class="modal-subtitle">Type a partial prompt to observe vocabulary logits, temperature scaling, and click bars to generate text.</p>
                    </div>
                    <button id="btn-close-prediction-modal" class="btn-icon-only modal-close">✕</button>
                </div>

                <div style="padding:16px 24px; background:#080f1c; border-bottom:1px solid rgba(255,255,255,0.06); display:flex; flex-direction:column; gap:12px;">
                    <div style="display:flex; gap:10px;">
                        <input id="prediction-prompt-input" class="modal-search-input" type="text" value="The transformer attention" style="flex:1; font-size:13px; font-weight:600;" placeholder="Enter partial sentence...">
                        <button id="btn-run-prediction" class="round-button btn-catalog" style="font-size:11px;">Predict ⚡</button>
                        <button id="btn-clear-prediction" class="round-button" style="font-size:11px;">Clear ✕</button>
                    </div>

                    <div style="display:flex; flex-wrap:wrap; gap:18px; align-items:center; font-size:11px;">
                        <div style="display:flex; align-items:center; gap:8px;">
                            <label style="color:var(--text-faint);">Temperature T:</label>
                            <input id="prediction-temp-slider" type="range" min="0.1" max="2.0" step="0.05" value="0.8" style="width:110px; cursor:pointer;">
                            <strong id="prediction-temp-val" style="color:var(--cyan); font-family:var(--font-mono); width:32px;">0.8</strong>
                        </div>

                        <div style="display:flex; align-items:center; gap:8px;">
                            <label style="color:var(--text-faint);">Top-K:</label>
                            <input id="prediction-topk-slider" type="range" min="1" max="30" step="1" value="10" style="width:100px; cursor:pointer;">
                            <strong id="prediction-topk-val" style="color:#f59e0b; font-family:var(--font-mono); width:24px;">10</strong>
                        </div>

                        <div style="display:flex; align-items:center; gap:6px;">
                            <button id="btn-mode-greedy" class="round-button active" style="font-size:10px; padding:3px 8px;">Greedy (Argmax)</button>
                            <button id="btn-mode-sampling" class="round-button" style="font-size:10px; padding:3px 8px;">Top-P Sampling</button>
                        </div>
                    </div>
                </div>

                <div style="flex:1; overflow-y:auto; padding:20px 24px; display:flex; flex-direction:column; gap:16px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-size:11px; font-weight:700; color:var(--text-primary); text-transform:uppercase; letter-spacing:0.5px;">
                            📊 Top-10 Predicted Candidate Tokens
                        </span>
                        <span style="font-size:10px; color:var(--text-faint);">Click any bar to append token and step forward</span>
                    </div>

                    <div id="prediction-bars-container" style="display:flex; flex-direction:column; gap:8px;"></div>
                </div>
            </div>
        `;
        document.body.appendChild(modal);

        modal.querySelector('#btn-close-prediction-modal').addEventListener('click', () => modal.classList.add('hidden'));
        modal.addEventListener('click', (e) => {
            if (e.target === modal) modal.classList.add('hidden');
        });
    }

    const inputEl = document.getElementById('prediction-prompt-input');
    const tempSlider = document.getElementById('prediction-temp-slider');
    const topkSlider = document.getElementById('prediction-topk-slider');
    const tempValEl = document.getElementById('prediction-temp-val');
    const topkValEl = document.getElementById('prediction-topk-val');
    const barsBox = document.getElementById('prediction-bars-container');

    function renderPrediction() {
        const prompt = inputEl.value;
        const temp = parseFloat(tempSlider.value) || 0.8;
        const topK = parseInt(topkSlider.value, 10) || 10;
        tempValEl.innerText = temp.toFixed(2);
        topkValEl.innerText = topK;

        const candidates = predictNextToken(prompt, temp, topK);
        const maxProb = candidates[0] ? candidates[0].probability : 1.0;

        barsBox.innerHTML = candidates.map((c, i) => {
            const pct = (c.probability * 100).toFixed(1);
            const barWidth = Math.max(8, Math.round((c.probability / (maxProb || 1.0)) * 100));
            const color = i === 0 ? '#34d399' : (i < 3 ? '#38bdf8' : '#818cf8');

            return `
                <div class="prediction-row" data-token="${c.token}" style="display:flex; align-items:center; gap:12px; background:rgba(6,13,24,0.7); border:1px solid rgba(255,255,255,0.06); border-radius:6px; padding:8px 12px; cursor:pointer; transition:all 0.15s ease;">
                    <div style="width:24px; font-size:11px; font-weight:700; color:var(--text-faint); font-family:var(--font-mono);">#${i + 1}</div>
                    <div style="width:110px; font-size:12px; font-weight:700; color:#fff; font-family:var(--font-mono); overflow:hidden; text-overflow:ellipsis;">"${c.token}"</div>
                    <div style="flex:1; background:rgba(255,255,255,0.05); height:12px; border-radius:6px; overflow:hidden;">
                        <div style="height:100%; width:${barWidth}%; background:${color}; border-radius:6px; transition:width 0.3s ease;"></div>
                    </div>
                    <div style="width:60px; text-align:right; font-size:11px; font-weight:700; color:${color}; font-family:var(--font-mono);">${pct}%</div>
                    <div style="width:60px; text-align:right; font-size:10px; color:var(--text-faint); font-family:var(--font-mono);">${c.logit.toFixed(2)}</div>
                </div>
            `;
        }).join('');

        barsBox.querySelectorAll('.prediction-row').forEach(row => {
            row.addEventListener('click', () => {
                const nextTok = row.dataset.token;
                inputEl.value = `${inputEl.value.trim()} ${nextTok}`;
                renderPrediction();
                if (onTokenSelected) onTokenSelected(nextTok, inputEl.value);
            });
        });
    }

    document.getElementById('btn-run-prediction').onclick = renderPrediction;
    document.getElementById('btn-clear-prediction').onclick = () => {
        inputEl.value = 'The transformer';
        renderPrediction();
    };
    tempSlider.oninput = renderPrediction;
    topkSlider.oninput = renderPrediction;
    inputEl.onkeydown = (e) => {
        if (e.key === 'Enter') renderPrediction();
    };

    modal.classList.remove('hidden');
    renderPrediction();
}
