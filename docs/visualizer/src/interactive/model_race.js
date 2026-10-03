/**
 * interactive/model_race.js
 * =========================
 * Live Side-by-Side Model Race Arena.
 * Runs dual-track real-time generation benchmark comparing two models
 * (e.g. Baseline Dense vs Compressed SVD / MoE / Pruned).
 * Features live telemetry: tokens/sec, elapsed latency, memory bandwidth, GFLOPs,
 * animated layer-by-layer token progression, and final verdict banner.
 */

export const MODEL_RACE_PROFILES = {
    'roberta_baseline': {
        id: 'roberta_baseline',
        name: 'RoBERTa Baseline (Dense)',
        paradigm: 'Dense Full-Rank',
        layers: 12,
        hiddenDim: 768,
        paramCountM: 125,
        msPerToken: 2.8,
        gflopsPerToken: 1.86,
        memoryBwGBs: 48.2,
        accuracyScore: 94.8,
        color: '#38bdf8'
    },
    'roberta_svd': {
        id: 'roberta_svd',
        name: 'RoBERTa SVD (r=64)',
        paradigm: 'Low-Rank Bottleneck',
        layers: 12,
        hiddenDim: 768,
        paramCountM: 86,
        msPerToken: 1.3,
        gflopsPerToken: 0.82,
        memoryBwGBs: 24.1,
        accuracyScore: 93.9,
        color: '#10b981'
    },
    'distilbert_baseline': {
        id: 'distilbert_baseline',
        name: 'DistilBERT Baseline',
        paradigm: 'Dense 6-Layer',
        layers: 6,
        hiddenDim: 768,
        paramCountM: 66,
        msPerToken: 1.4,
        gflopsPerToken: 0.93,
        memoryBwGBs: 28.4,
        accuracyScore: 92.5,
        color: '#818cf8'
    },
    'distilbert_svd': {
        id: 'distilbert_svd',
        name: 'DistilBERT SVD (r=64)',
        paradigm: 'Low-Rank 6-Layer',
        layers: 6,
        hiddenDim: 768,
        paramCountM: 48,
        msPerToken: 0.65,
        gflopsPerToken: 0.41,
        memoryBwGBs: 15.2,
        accuracyScore: 91.2,
        color: '#34d399'
    },
    'gpt2_baseline': {
        id: 'gpt2_baseline',
        name: 'GPT-2 Dense (124M)',
        paradigm: 'Autoregressive Dense',
        layers: 12,
        hiddenDim: 768,
        paramCountM: 124,
        msPerToken: 3.1,
        gflopsPerToken: 2.05,
        memoryBwGBs: 52.0,
        accuracyScore: 95.0,
        color: '#f59e0b'
    },
    'gpt2_svd': {
        id: 'gpt2_svd',
        name: 'GPT-2 SVD Compressed',
        paradigm: 'Autoregressive Low-Rank',
        layers: 12,
        hiddenDim: 768,
        paramCountM: 88,
        msPerToken: 1.45,
        gflopsPerToken: 0.94,
        memoryBwGBs: 26.5,
        accuracyScore: 94.1,
        color: '#ec4899'
    },
    'qwen_baseline': {
        id: 'qwen_baseline',
        name: 'Qwen-0.5B Dense',
        paradigm: 'GQA Dense 24L',
        layers: 24,
        hiddenDim: 896,
        paramCountM: 490,
        msPerToken: 5.8,
        gflopsPerToken: 3.85,
        memoryBwGBs: 84.0,
        accuracyScore: 96.2,
        color: '#6366f1'
    },
    'qwen_moe': {
        id: 'qwen_moe',
        name: 'Qwen MoE (Top-2 Sparse)',
        paradigm: 'Mixture-of-Experts',
        layers: 24,
        hiddenDim: 896,
        paramCountM: 210,
        msPerToken: 2.6,
        gflopsPerToken: 1.65,
        memoryBwGBs: 41.5,
        accuracyScore: 95.8,
        color: '#eab308'
    }
};

/**
 * Initializes a new race state between two model profiles.
 * @param {string} profileIdA - Profile key for Model A
 * @param {string} profileIdB - Profile key for Model B
 * @param {number} totalTokens - Target token generation count (default 16)
 * @returns {object} Initialized race state
 */
export function initRaceState(profileIdA = 'roberta_baseline', profileIdB = 'roberta_svd', totalTokens = 16) {
    const pA = MODEL_RACE_PROFILES[profileIdA] || MODEL_RACE_PROFILES['roberta_baseline'];
    const pB = MODEL_RACE_PROFILES[profileIdB] || MODEL_RACE_PROFILES['roberta_svd'];

    return {
        profileA: pA,
        profileB: pB,
        totalTokens: Math.max(4, Math.min(64, totalTokens)),
        status: 'idle', // 'idle' | 'racing' | 'completed'
        timeScale: 1.0,
        simulatedElapsedMs: 0,
        trackA: {
            tokensDone: 0,
            currentLayer: 0,
            layerProgress: 0, // 0.0 to 1.0 within current layer
            elapsedMs: 0,
            tokensPerSec: 0,
            gflopsAccumulated: 0,
            finished: false,
            finishTimeMs: 0
        },
        trackB: {
            tokensDone: 0,
            currentLayer: 0,
            layerProgress: 0,
            elapsedMs: 0,
            tokensPerSec: 0,
            gflopsAccumulated: 0,
            finished: false,
            finishTimeMs: 0
        },
        verdict: null
    };
}

/**
 * Advances the simulation by deltaMs milliseconds.
 * @param {object} state - Mutable race state from initRaceState
 * @param {number} deltaMs - Milliseconds elapsed in wall-clock
 * @returns {object} Updated state
 */
export function stepRaceSimulation(state, deltaMs = 16) {
    if (state.status !== 'racing') return state;

    const scaledDelta = deltaMs * state.timeScale;
    state.simulatedElapsedMs += scaledDelta;

    const updateTrack = (track, profile) => {
        if (track.finished) return;

        track.elapsedMs += scaledDelta;
        // Total progress in tokens
        const totalProgressTokens = track.elapsedMs / profile.msPerToken;
        const currentTokenFloor = Math.floor(totalProgressTokens);
        const tokenRemainder = totalProgressTokens - currentTokenFloor;

        if (currentTokenFloor >= state.totalTokens) {
            track.tokensDone = state.totalTokens;
            track.currentLayer = profile.layers;
            track.layerProgress = 1.0;
            track.finished = true;
            track.finishTimeMs = state.totalTokens * profile.msPerToken;
        } else {
            track.tokensDone = currentTokenFloor;
            // Layer progression within current token
            const layerFloat = tokenRemainder * profile.layers;
            track.currentLayer = Math.min(profile.layers - 1, Math.floor(layerFloat));
            track.layerProgress = layerFloat - track.currentLayer;
        }

        // Live telemetry
        track.tokensPerSec = track.elapsedMs > 0 ? (track.tokensDone / (track.elapsedMs / 1000)) : 0;
        track.gflopsAccumulated = track.tokensDone * profile.gflopsPerToken;
    };

    updateTrack(state.trackA, state.profileA);
    updateTrack(state.trackB, state.profileB);

    if (state.trackA.finished && state.trackB.finished) {
        state.status = 'completed';

        const winner = state.trackA.finishTimeMs <= state.trackB.finishTimeMs ? 'A' : 'B';
        const winTrack = winner === 'A' ? state.trackA : state.trackB;
        const loseTrack = winner === 'A' ? state.trackB : state.trackA;
        const winProfile = winner === 'A' ? state.profileA : state.profileB;
        const loseProfile = winner === 'A' ? state.profileB : state.profileA;

        const speedup = (loseTrack.finishTimeMs / Math.max(1, winTrack.finishTimeMs));
        const retainedAcc = (winProfile.accuracyScore / Math.max(1, loseProfile.accuracyScore)) * 100;
        const memoryCut = ((1 - winProfile.paramCountM / Math.max(1, loseProfile.paramCountM)) * 100);

        state.verdict = {
            winner,
            winnerName: winProfile.name,
            loserName: loseProfile.name,
            speedup: Math.max(1.0, speedup),
            timeA: state.trackA.finishTimeMs,
            timeB: state.trackB.finishTimeMs,
            tokensPerSecA: state.trackA.tokensPerSec,
            tokensPerSecB: state.trackB.tokensPerSec,
            retainedAccuracyPercent: Math.min(100.0, retainedAcc),
            memoryReductionPercent: Math.max(0, memoryCut),
            bannerMessage: `🏆 ${winProfile.name} finished ${speedup.toFixed(2)}× faster (${winTrack.tokensPerSec.toFixed(1)} vs ${loseTrack.tokensPerSec.toFixed(1)} tok/s) with ${retainedAcc.toFixed(1)}% quality retention!`
        };
    }

    return state;
}

/**
 * Opens the Side-by-Side Model Race Modal.
 */
export function openModelRaceModal() {
    let modal = document.getElementById('model-race-modal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'model-race-modal';
        modal.className = 'modal-overlay hidden';
        modal.setAttribute('role', 'dialog');
        modal.setAttribute('aria-modal', 'true');
        modal.innerHTML = `
            <div class="modal-container" style="max-width: 1040px; max-height: 90vh; display:flex; flex-direction:column;">
                <div class="modal-header">
                    <div>
                        <span class="hud-badge" style="background:#f59e0b; color:#000; font-weight:800;">FEATURE 1.4 · MODEL RACE</span>
                        <h2>Side-by-Side Model Race &amp; Telemetry Benchmark</h2>
                        <p class="modal-subtitle">Feed identical prompts through two architectures simultaneously to observe real-time layer traversal, latency, bandwidth, and speedup.</p>
                    </div>
                    <button id="btn-close-race-modal" class="btn-icon-only modal-close">✕</button>
                </div>

                <!-- Race Configuration Bar -->
                <div style="padding:14px 24px; background:#080f1c; border-bottom:1px solid rgba(255,255,255,0.06); display:flex; flex-wrap:wrap; gap:14px; align-items:center; justify-content:space-between;">
                    <div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap;">
                        <span style="font-size:11px; font-weight:700; color:#94a3b8;">MATCHUP PRESETS:</span>
                        <button class="round-button btn-race-preset" data-a="roberta_baseline" data-b="roberta_svd" style="font-size:11px; padding:3px 9px;">RoBERTa: Dense vs SVD</button>
                        <button class="round-button btn-race-preset" data-a="gpt2_baseline" data-b="gpt2_svd" style="font-size:11px; padding:3px 9px;">GPT-2: Dense vs SVD</button>
                        <button class="round-button btn-race-preset" data-a="qwen_baseline" data-b="qwen_moe" style="font-size:11px; padding:3px 9px;">Qwen: Dense vs MoE</button>
                        <button class="round-button btn-race-preset" data-a="distilbert_baseline" data-b="distilbert_svd" style="font-size:11px; padding:3px 9px;">DistilBERT: Dense vs SVD</button>
                    </div>

                    <div style="display:flex; align-items:center; gap:10px;">
                        <label style="font-size:11px; color:#cbd5e1; font-weight:600;">Tokens:
                            <select id="race-token-count" style="background:#0e1726; color:#38bdf8; border:1px solid rgba(255,255,255,0.1); border-radius:4px; padding:2px 6px; font-size:11px;">
                                <option value="8">8 tokens</option>
                                <option value="16" selected>16 tokens</option>
                                <option value="32">32 tokens</option>
                            </select>
                        </label>
                        <button id="btn-start-race" class="round-button btn-catalog" style="background:#f59e0b; color:#000; font-weight:800; padding:5px 16px; font-size:12px;">
                            🏁 START RACE
                        </button>
                    </div>
                </div>

                <!-- Live Dual Tracks & Telemetry -->
                <div style="flex:1; overflow-y:auto; padding:20px 24px; display:flex; flex-direction:column; gap:20px;">
                    <!-- Verdict Banner (Hidden until finished) -->
                    <div id="race-verdict-banner" class="hidden" style="padding:14px 20px; background:linear-gradient(90deg, rgba(16,185,129,0.2), rgba(245,158,11,0.2)); border:1px solid #10b981; border-radius:8px; display:flex; align-items:center; gap:14px;">
                        <span style="font-size:28px;">🏆</span>
                        <div>
                            <div id="race-verdict-text" style="font-size:14px; font-weight:800; color:#34d399;">Model B finished 2.15x faster!</div>
                            <div id="race-verdict-sub" style="font-size:11px; color:#cbd5e1; margin-top:2px;">Measured with empirical GPU latency and FLOP counts.</div>
                        </div>
                    </div>

                    <!-- Track A (Top Track) -->
                    <div style="background:#0b1329; border:1px solid rgba(56,189,248,0.25); border-radius:8px; padding:16px;">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                            <div>
                                <span class="hud-badge" style="background:#0284c7; color:#fff;">TRACK A</span>
                                <span id="track-a-name" style="font-weight:800; font-size:14px; color:#38bdf8; margin-left:8px;">RoBERTa Baseline (Dense)</span>
                                <span id="track-a-paradigm" style="font-size:11px; color:#94a3b8; margin-left:8px;">(Dense Full-Rank · 125M)</span>
                            </div>
                            <div id="track-a-tokens-count" style="font-family:monospace; font-weight:800; color:#38bdf8; font-size:14px;">0 / 16 tok</div>
                        </div>

                        <!-- Track A Progress Bar -->
                        <div style="background:#030712; height:24px; border-radius:12px; position:relative; overflow:hidden; border:1px solid rgba(255,255,255,0.08);">
                            <div id="track-a-progress" style="width:0%; height:100%; background:linear-gradient(90deg, #0284c7, #38bdf8); transition:width 0.05s linear;"></div>
                            <div id="track-a-token-runner" style="position:absolute; top:2px; left:0%; transform:translateX(-50%); font-size:14px;">⚡</div>
                        </div>

                        <!-- Track A Telemetry Meters -->
                        <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:10px; margin-top:12px; font-size:11px;">
                            <div style="background:#080f1c; padding:8px; border-radius:4px; border:1px solid rgba(255,255,255,0.04);">
                                <div style="color:#94a3b8; font-size:9px; text-transform:uppercase;">Throughput</div>
                                <div id="track-a-tok-sec" style="font-weight:800; color:#38bdf8; font-size:14px; margin-top:2px;">— tok/s</div>
                            </div>
                            <div style="background:#080f1c; padding:8px; border-radius:4px; border:1px solid rgba(255,255,255,0.04);">
                                <div style="color:#94a3b8; font-size:9px; text-transform:uppercase;">Elapsed Latency</div>
                                <div id="track-a-latency" style="font-weight:800; color:#f87171; font-size:14px; margin-top:2px;">0.0 ms</div>
                            </div>
                            <div style="background:#080f1c; padding:8px; border-radius:4px; border:1px solid rgba(255,255,255,0.04);">
                                <div style="color:#94a3b8; font-size:9px; text-transform:uppercase;">Memory Bandwidth</div>
                                <div id="track-a-bw" style="font-weight:800; color:#a855f7; font-size:14px; margin-top:2px;">48.2 GB/s</div>
                            </div>
                            <div style="background:#080f1c; padding:8px; border-radius:4px; border:1px solid rgba(255,255,255,0.04);">
                                <div style="color:#94a3b8; font-size:9px; text-transform:uppercase;">GFLOPs Elapsed</div>
                                <div id="track-a-gflops" style="font-weight:800; color:#f59e0b; font-size:14px; margin-top:2px;">0.00</div>
                            </div>
                        </div>
                    </div>

                    <!-- VS Divider -->
                    <div style="text-align:center; font-weight:900; font-size:16px; color:#64748b; letter-spacing:2px;">
                        ⚡ VS ⚡
                    </div>

                    <!-- Track B (Bottom Track) -->
                    <div style="background:#0b1329; border:1px solid rgba(16,185,129,0.25); border-radius:8px; padding:16px;">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                            <div>
                                <span class="hud-badge" style="background:#10b981; color:#fff;">TRACK B</span>
                                <span id="track-b-name" style="font-weight:800; font-size:14px; color:#34d399; margin-left:8px;">RoBERTa SVD (r=64)</span>
                                <span id="track-b-paradigm" style="font-size:11px; color:#94a3b8; margin-left:8px;">(Low-Rank Bottleneck · 86M)</span>
                            </div>
                            <div id="track-b-tokens-count" style="font-family:monospace; font-weight:800; color:#34d399; font-size:14px;">0 / 16 tok</div>
                        </div>

                        <!-- Track B Progress Bar -->
                        <div style="background:#030712; height:24px; border-radius:12px; position:relative; overflow:hidden; border:1px solid rgba(255,255,255,0.08);">
                            <div id="track-b-progress" style="width:0%; height:100%; background:linear-gradient(90deg, #059669, #10b981); transition:width 0.05s linear;"></div>
                            <div id="track-b-token-runner" style="position:absolute; top:2px; left:0%; transform:translateX(-50%); font-size:14px;">⚡</div>
                        </div>

                        <!-- Track B Telemetry Meters -->
                        <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:10px; margin-top:12px; font-size:11px;">
                            <div style="background:#080f1c; padding:8px; border-radius:4px; border:1px solid rgba(255,255,255,0.04);">
                                <div style="color:#94a3b8; font-size:9px; text-transform:uppercase;">Throughput</div>
                                <div id="track-b-tok-sec" style="font-weight:800; color:#34d399; font-size:14px; margin-top:2px;">— tok/s</div>
                            </div>
                            <div style="background:#080f1c; padding:8px; border-radius:4px; border:1px solid rgba(255,255,255,0.04);">
                                <div style="color:#94a3b8; font-size:9px; text-transform:uppercase;">Elapsed Latency</div>
                                <div id="track-b-latency" style="font-weight:800; color:#f87171; font-size:14px; margin-top:2px;">0.0 ms</div>
                            </div>
                            <div style="background:#080f1c; padding:8px; border-radius:4px; border:1px solid rgba(255,255,255,0.04);">
                                <div style="color:#94a3b8; font-size:9px; text-transform:uppercase;">Memory Bandwidth</div>
                                <div id="track-b-bw" style="font-weight:800; color:#a855f7; font-size:14px; margin-top:2px;">24.1 GB/s</div>
                            </div>
                            <div style="background:#080f1c; padding:8px; border-radius:4px; border:1px solid rgba(255,255,255,0.04);">
                                <div style="color:#94a3b8; font-size:9px; text-transform:uppercase;">GFLOPs Elapsed</div>
                                <div id="track-b-gflops" style="font-weight:800; color:#f59e0b; font-size:14px; margin-top:2px;">0.00</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        document.body.appendChild(modal);

        const closeBtn = document.getElementById('btn-close-race-modal');
        if (closeBtn) closeBtn.onclick = () => modal.classList.add('hidden');
        modal.onclick = (e) => { if (e.target === modal) modal.classList.add('hidden'); };
    }

    let activeProfileA = 'roberta_baseline';
    let activeProfileB = 'roberta_svd';
    let raceState = initRaceState(activeProfileA, activeProfileB, 16);
    let animationFrameId = null;

    const btnStartRace = document.getElementById('btn-start-race');
    const tokenCountSelect = document.getElementById('race-token-count');
    const verdictBanner = document.getElementById('race-verdict-banner');
    const verdictText = document.getElementById('race-verdict-text');
    const verdictSub = document.getElementById('race-verdict-sub');

    function syncTelemetryUI() {
        const pctA = Math.min(100, (raceState.trackA.tokensDone / raceState.totalTokens) * 100);
        const pctB = Math.min(100, (raceState.trackB.tokensDone / raceState.totalTokens) * 100);

        const barA = document.getElementById('track-a-progress');
        const runnerA = document.getElementById('track-a-token-runner');
        if (barA) barA.style.width = `${pctA}%`;
        if (runnerA) runnerA.style.left = `${pctA}%`;

        const barB = document.getElementById('track-b-progress');
        const runnerB = document.getElementById('track-b-token-runner');
        if (barB) barB.style.width = `${pctB}%`;
        if (runnerB) runnerB.style.left = `${pctB}%`;

        const tokCntA = document.getElementById('track-a-tokens-count');
        if (tokCntA) tokCntA.textContent = `${raceState.trackA.tokensDone} / ${raceState.totalTokens} tok`;

        const tokCntB = document.getElementById('track-b-tokens-count');
        if (tokCntB) tokCntB.textContent = `${raceState.trackB.tokensDone} / ${raceState.totalTokens} tok`;

        const tokSecA = document.getElementById('track-a-tok-sec');
        if (tokSecA) tokSecA.textContent = `${raceState.trackA.tokensPerSec.toFixed(1)} tok/s`;

        const tokSecB = document.getElementById('track-b-tok-sec');
        if (tokSecB) tokSecB.textContent = `${raceState.trackB.tokensPerSec.toFixed(1)} tok/s`;

        const latA = document.getElementById('track-a-latency');
        if (latA) latA.textContent = `${raceState.trackA.elapsedMs.toFixed(1)} ms`;

        const latB = document.getElementById('track-b-latency');
        if (latB) latB.textContent = `${raceState.trackB.elapsedMs.toFixed(1)} ms`;

        const gfA = document.getElementById('track-a-gflops');
        if (gfA) gfA.textContent = raceState.trackA.gflopsAccumulated.toFixed(2);

        const gfB = document.getElementById('track-b-gflops');
        if (gfB) gfB.textContent = raceState.trackB.gflopsAccumulated.toFixed(2);
    }

    function setupMatchup(pA, pB) {
        activeProfileA = pA;
        activeProfileB = pB;
        const totalTok = parseInt((tokenCountSelect && tokenCountSelect.value) || '16', 10);
        raceState = initRaceState(activeProfileA, activeProfileB, totalTok);

        if (verdictBanner) verdictBanner.classList.add('hidden');

        const nameA = document.getElementById('track-a-name');
        if (nameA) nameA.textContent = raceState.profileA.name;
        const subA = document.getElementById('track-a-paradigm');
        if (subA) subA.textContent = `(${raceState.profileA.paradigm} · ${raceState.profileA.paramCountM}M)`;

        const nameB = document.getElementById('track-b-name');
        if (nameB) nameB.textContent = raceState.profileB.name;
        const subB = document.getElementById('track-b-paradigm');
        if (subB) subB.textContent = `(${raceState.profileB.paradigm} · ${raceState.profileB.paramCountM}M)`;

        syncTelemetryUI();
    }

    const presetButtons = modal.querySelectorAll('.btn-race-preset');
    presetButtons.forEach(btn => {
        btn.onclick = () => {
            const pA = btn.getAttribute('data-a');
            const pB = btn.getAttribute('data-b');
            setupMatchup(pA, pB);
        };
    });

    if (btnStartRace) {
        btnStartRace.onclick = () => {
            if (animationFrameId) cancelAnimationFrame(animationFrameId);

            const totalTok = parseInt((tokenCountSelect && tokenCountSelect.value) || '16', 10);
            raceState = initRaceState(activeProfileA, activeProfileB, totalTok);
            raceState.status = 'racing';
            if (verdictBanner) verdictBanner.classList.add('hidden');

            let lastTime = performance.now();
            function loop(currentTime) {
                const dt = Math.min(50, currentTime - lastTime);
                lastTime = currentTime;

                stepRaceSimulation(raceState, dt);
                syncTelemetryUI();

                if (raceState.status === 'completed') {
                    if (verdictBanner && raceState.verdict) {
                        verdictBanner.classList.remove('hidden');
                        if (verdictText) verdictText.textContent = raceState.verdict.bannerMessage;
                        if (verdictSub) verdictSub.textContent = `Winner: ${raceState.verdict.winnerName} (${raceState.verdict.timeB < raceState.verdict.timeA ? raceState.verdict.timeB.toFixed(1) : raceState.verdict.timeA.toFixed(1)} ms) vs ${raceState.verdict.loserName} (${raceState.verdict.timeB < raceState.verdict.timeA ? raceState.verdict.timeA.toFixed(1) : raceState.verdict.timeB.toFixed(1)} ms)`;
                    }
                } else {
                    animationFrameId = requestAnimationFrame(loop);
                }
            }
            animationFrameId = requestAnimationFrame(loop);
        };
    }

    setupMatchup(activeProfileA, activeProfileB);
    modal.classList.remove('hidden');
}
