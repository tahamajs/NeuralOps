/**
 * educational/present_mode.js
 * ===========================
 * Fullscreen Cinematic Presentation Mode (5-Minute Demo Ready).
 * Minimal interface with:
 * - Semi-transparent glassmorphic inspector drawer
 * - 7-Stage Token Progress Bar at the bottom
 * - Smooth camera transitions and slower, high-elegance particle physics
 * - Escape key to exit seamlessly
 */

import { ATTENTION_TOUR_STEPS } from './attention_tour.js';

let isPresenting = false;
let presentStep = 0;
let presentTimer = null;

/**
 * Toggles Presentation Mode on and off.
 * @param {object} context - { camera, controls, store, rebuildSceneForModel }
 */
export function togglePresentMode(context = {}) {
    isPresenting = !isPresenting;
    const body = document.body;
    let bar = document.getElementById('present-dock-bar');

    if (!bar) {
        bar = document.createElement('div');
        bar.id = 'present-dock-bar';
        bar.className = 'present-dock-bar hidden';
        bar.innerHTML = `
            <div class="present-banner" style="position:fixed; top:18px; left:50%; transform:translateX(-50%); z-index:9999; background:rgba(6,13,24,0.85); backdrop-filter:blur(16px); border:1px solid rgba(133,198,235,0.3); border-radius:30px; padding:8px 24px; display:flex; align-items:center; gap:16px; box-shadow:0 12px 40px rgba(0,0,0,0.6);">
                <span class="badge" style="background:rgba(16,185,129,0.2); color:#34d399; font-weight:700;">▶ 5-MIN DEMO PRESENTATION</span>
                <span id="present-step-title" style="font-size:13px; font-weight:600; color:#eef8ff;">Welcome to Attention Lab</span>
                <div style="display:flex; gap:8px;">
                    <button id="btn-present-prev" class="round-button" style="padding:2px 10px; font-size:11px;">◀ Prev</button>
                    <button id="btn-present-next" class="round-button" style="padding:2px 10px; font-size:11px; background:rgba(34,213,247,0.2); color:#38bdf8;">Next ▶</button>
                    <button id="btn-present-exit" class="round-button" style="padding:2px 10px; font-size:11px; color:#f87171;">✕ Exit (Esc)</button>
                </div>
            </div>
            <div class="present-progress-track" style="position:fixed; bottom:20px; left:50%; transform:translateX(-50%); z-index:9999; background:rgba(6,13,24,0.85); backdrop-filter:blur(16px); border:1px solid rgba(133,198,235,0.25); border-radius:24px; padding:10px 24px; display:flex; gap:12px; align-items:center;">
                ${ATTENTION_TOUR_STEPS.map((s, i) => `
                    <div class="present-dot ${i === 0 ? 'active' : ''}" data-step="${i}" style="width:12px; height:12px; border-radius:50%; background:${i === 0 ? 'var(--cyan)' : 'rgba(255,255,255,0.2)'}; cursor:pointer; transition:all 0.3s;" title="${s.title}"></div>
                `).join('')}
            </div>
        `;
        document.body.appendChild(bar);

        bar.querySelector('#btn-present-prev').addEventListener('click', () => stepPresent(-1, context));
        bar.querySelector('#btn-present-next').addEventListener('click', () => stepPresent(1, context));
        bar.querySelector('#btn-present-exit').addEventListener('click', () => togglePresentMode(context));

        bar.querySelectorAll('.present-dot').forEach(dot => {
            dot.addEventListener('click', () => {
                presentStep = parseInt(dot.dataset.step, 10);
                showStep(presentStep, context);
            });
        });
    }

    if (isPresenting) {
        body.classList.add('present-mode');
        bar.classList.remove('hidden');
        presentStep = 0;
        showStep(presentStep, context);
        try {
            if (document.documentElement.requestFullscreen) {
                document.documentElement.requestFullscreen().catch(() => {});
            }
        } catch (e) {}
    } else {
        body.classList.remove('present-mode');
        bar.classList.add('hidden');
        clearTimeout(presentTimer);
        try {
            if (document.exitFullscreen && document.fullscreenElement) {
                document.exitFullscreen().catch(() => {});
            }
        } catch (e) {}
    }
}

function stepPresent(delta, context) {
    presentStep = (presentStep + delta + ATTENTION_TOUR_STEPS.length) % ATTENTION_TOUR_STEPS.length;
    showStep(presentStep, context);
}

function showStep(stepIdx, context) {
    const s = ATTENTION_TOUR_STEPS[stepIdx];
    if (!s) return;

    const titleEl = document.getElementById('present-step-title');
    if (titleEl) titleEl.innerText = `${s.step}/10 · ${s.title}`;

    // Update Dots
    document.querySelectorAll('.present-dot').forEach((dot, i) => {
        dot.style.background = (i === stepIdx) ? 'var(--cyan)' : 'rgba(255,255,255,0.2)';
        dot.style.transform = (i === stepIdx) ? 'scale(1.4)' : 'scale(1.0)';
    });

    // Fly Camera
    if (context.camera && context.controls && window.smoothFlyCamera) {
        window.smoothFlyCamera(s.y, s.dist);
    }

    // Select Part in 3D
    if (s.target && window.selectPartIn3D) {
        window.selectPartIn3D(s.target);
    }
}

// Escape key listener to exit Presentation Mode
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && isPresenting) {
        togglePresentMode();
    }
});
