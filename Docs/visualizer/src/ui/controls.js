/**
 * ui/controls.js
 * ==============
 * Playback, timeline scrubbing, speed control, view mode toggle, and keyboard shortcuts.
 */

/**
 * Initializes all bottom toolbar and topbar user controls.
 * @param {object} handlers - Event callbacks { onPlayPause, onSpeedChange, onScrub, onSwitchView, onToggleInspector, onResetCamera, onOpenCatalog }
 */
export function setupControls(handlers = {}) {
    // 1. Play / Pause
    const playBtn = document.getElementById('btn-play-pause');
    if (playBtn) {
        playBtn.addEventListener('click', () => {
            const isPlaying = handlers.onPlayPause ? handlers.onPlayPause() : false;
            playBtn.innerText = isPlaying ? '⏸ Pause' : '▶ Play';
        });
    }

    // 2. Simulation Speed Multiplier
    const speedSelect = document.getElementById('select-speed');
    if (speedSelect) {
        speedSelect.addEventListener('change', (e) => {
            const val = parseFloat(e.target.value) || 1.0;
            if (handlers.onSpeedChange) handlers.onSpeedChange(val);
        });
    }

    // 3. Timeline Scrubber
    const scrubber = document.getElementById('timeline-scrubber');
    const stepLabel = document.getElementById('timeline-step');
    if (scrubber) {
        scrubber.addEventListener('input', (e) => {
            const step = parseInt(e.target.value, 10);
            if (stepLabel) stepLabel.innerText = `0${step + 1} Stage`;
            if (handlers.onScrub) handlers.onScrub(step);
        });
    }

    // 4. 2D / 3D Perspective Toggle
    const btn3d = document.getElementById('btn-view-3d');
    const btn2d = document.getElementById('btn-view-2d');

    if (btn3d && btn2d) {
        btn3d.addEventListener('click', () => {
            btn3d.classList.add('active');
            btn2d.classList.remove('active');
            if (handlers.onSwitchView) handlers.onSwitchView('3d');
        });
        btn2d.addEventListener('click', () => {
            btn2d.classList.add('active');
            btn3d.classList.remove('active');
            if (handlers.onSwitchView) handlers.onSwitchView('2d');
        });
    }

    // 5. Deep Inspector Drawer Toggle
    const inspBtn = document.getElementById('btn-toggle-inspector');
    const closeInspBtn = document.getElementById('btn-close-insp');
    const inspector = document.getElementById('inspector');

    if (inspBtn && inspector) {
        inspBtn.addEventListener('click', () => {
            inspector.classList.toggle('hidden');
        });
    }
    if (closeInspBtn && inspector) {
        closeInspBtn.addEventListener('click', () => {
            inspector.classList.add('hidden');
        });
    }

    // 6. Keyboard Shortcuts
    window.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

        if (e.code === 'Space') {
            e.preventDefault();
            if (playBtn) playBtn.click();
        } else if (e.code === 'KeyR') {
            if (handlers.onResetCamera) handlers.onResetCamera();
        } else if (e.code === 'KeyM') {
            const catBtn = document.getElementById('btn-open-catalog');
            if (catBtn) catBtn.click();
        } else if (e.code === 'Escape') {
            if (inspector) inspector.classList.add('hidden');
            const modal = document.getElementById('module-modal');
            if (modal) modal.classList.add('hidden');
        }
    });
}
