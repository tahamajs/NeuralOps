/**
 * export/video_exporter.js
 * ========================
 * High-definition Canvas Video Recording and 360° Orbit Loop Exporter.
 * Leverages MediaRecorder API and HTMLCanvasElement.captureStream to capture
 * smooth 5-second or 10-second cinematic showcase loops with continuous token physics.
 */

/**
 * Computes exact 3D camera coordinates for a 360-degree orbit around a center point.
 * @param {object} center - Pivot center { x, y, z }
 * @param {number} radius - Orbit radius distance
 * @param {number} height - Elevation offset Y
 * @param {number} angleRad - Current rotation angle in radians (0 to 2*PI)
 * @returns {object} Calculated camera position { x, y, z }
 */
export function calculateOrbitFrameCameraPosition(center = { x: 0, y: 0, z: 0 }, radius = 22.0, height = 12.0, angleRad = 0.0) {
    return {
        x: (center.x || 0) + radius * Math.sin(angleRad),
        y: (center.y || 0) + height,
        z: (center.z || 0) + radius * Math.cos(angleRad)
    };
}

/**
 * Generates an array of discrete orbit keyframe positions for totalFrames.
 * @param {object} center - Pivot center
 * @param {number} radius - Orbit radius
 * @param {number} height - Elevation offset
 * @param {number} totalFrames - Total number of discrete frames
 * @returns {Array<object>} Array of { frame, angleRad, position }
 */
export function generateOrbitKeyframes(center = { x: 0, y: 0, z: 0 }, radius = 22.0, height = 12.0, totalFrames = 150) {
    const frames = Math.max(10, Math.round(totalFrames));
    const keyframes = [];

    for (let f = 0; f < frames; f++) {
        const tau = f / frames; // 0.0 to <1.0
        const angleRad = tau * 2 * Math.PI;
        keyframes.push({
            frame: f,
            tau,
            angleRad,
            position: calculateOrbitFrameCameraPosition(center, radius, height, angleRad)
        });
    }

    return keyframes;
}

/**
 * Validates recording parameters and calculates target frame counts and bitrates.
 * @param {number} durationSec - Desired duration in seconds (e.g. 5 or 10)
 * @param {number} fps - Framerate (e.g. 30 or 60)
 * @returns {object} Sanitized export configuration
 */
export function validateVideoExportConfig(durationSec = 5, fps = 30) {
    const d = Math.max(2, Math.min(30, Number(durationSec) || 5));
    const f = Math.max(15, Math.min(60, Number(fps) || 30));
    const totalFrames = Math.round(d * f);
    const frameIntervalMs = 1000 / f;
    const videoBitsPerSecond = 8000000; // 8 Mbps high quality

    return {
        durationSec: d,
        fps: f,
        totalFrames,
        frameIntervalMs,
        videoBitsPerSecond,
        defaultFilename: `attention_lab_${d}s_orbit_${Date.now()}.webm`
    };
}

/**
 * Executes canvas orbit recording in browser.
 * @param {object} options - Recording options { canvas, camera, controls, durationSec, fps, onProgress }
 * @returns {Promise<Blob>} Recorded video Blob
 */
export function recordCanvasOrbitLoop({
    canvas,
    camera,
    controls,
    durationSec = 5,
    fps = 30,
    onProgress = () => {}
}) {
    return new Promise((resolve, reject) => {
        if (!canvas || typeof canvas.captureStream !== 'function') {
            return reject(new Error('HTMLCanvasElement.captureStream is not supported in this browser environment.'));
        }

        if (typeof MediaRecorder === 'undefined') {
            return reject(new Error('MediaRecorder API is not available in this environment.'));
        }

        const config = validateVideoExportConfig(durationSec, fps);
        const stream = canvas.captureStream(config.fps);

        const mimeTypes = [
            'video/webm;codecs=vp9',
            'video/webm;codecs=vp8',
            'video/webm',
            'video/mp4'
        ];
        let chosenMime = mimeTypes.find(m => MediaRecorder.isTypeSupported(m)) || 'video/webm';

        let recorder;
        try {
            recorder = new MediaRecorder(stream, {
                mimeType: chosenMime,
                videoBitsPerSecond: config.videoBitsPerSecond
            });
        } catch (e) {
            recorder = new MediaRecorder(stream);
        }

        const recordedChunks = [];
        recorder.ondataavailable = (e) => {
            if (e.data && e.data.size > 0) {
                recordedChunks.push(e.data);
            }
        };

        const targetCenter = controls ? controls.target.clone() : { x: 0, y: 12, z: 0 };
        const initialPos = camera ? camera.position.clone() : { x: 0, y: 12, z: 22 };
        const radius = Math.hypot(initialPos.x - targetCenter.x, initialPos.z - targetCenter.z) || 22.0;
        const height = initialPos.y - targetCenter.y || 8.0;

        const startTime = performance.now();
        const totalDurationMs = config.durationSec * 1000;
        let animationFrameId = null;

        recorder.onstop = () => {
            if (animationFrameId) cancelAnimationFrame(animationFrameId);
            onProgress({ stage: 'encoding', percent: 90, message: 'Encoding WebM container...' });

            const blob = new Blob(recordedChunks, { type: chosenMime });
            onProgress({ stage: 'ready', percent: 100, message: 'Video Ready! Downloading...' });

            // Trigger download
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = config.defaultFilename;
            document.body.appendChild(a);
            a.click();
            setTimeout(() => {
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            }, 1000);

            resolve(blob);
        };

        // Start recording
        recorder.start(100); // chunk every 100ms
        onProgress({ stage: 'recording', percent: 0, message: 'Recording 360° Orbit Loop...' });

        function orbitLoop(now) {
            const elapsed = now - startTime;
            const tau = Math.min(1.0, elapsed / totalDurationMs);
            const percent = Math.round(tau * 85); // 0 to 85% is recording

            onProgress({
                stage: 'recording',
                percent,
                message: `Recording 360° Orbit: ${percent}% (${(elapsed / 1000).toFixed(1)}s / ${config.durationSec}s)`
            });

            // Smooth 360 degree rotation
            if (camera && controls) {
                const angle = tau * 2 * Math.PI;
                const newPos = calculateOrbitFrameCameraPosition(targetCenter, radius, height, angle);
                camera.position.set(newPos.x, newPos.y, newPos.z);
                camera.lookAt(targetCenter);
            }

            if (tau < 1.0) {
                animationFrameId = requestAnimationFrame(orbitLoop);
            } else {
                recorder.stop();
            }
        }

        animationFrameId = requestAnimationFrame(orbitLoop);
    });
}

/**
 * Displays the Video Export Modal.
 * @param {object} context - Engine context { canvas, camera, controls }
 */
export function openVideoExportModal({ canvas, camera, controls }) {
    let modal = document.getElementById('video-export-modal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'video-export-modal';
        modal.className = 'modal-overlay hidden';
        modal.setAttribute('role', 'dialog');
        modal.setAttribute('aria-modal', 'true');
        modal.innerHTML = `
            <div class="modal-container" style="max-width: 600px; display:flex; flex-direction:column;">
                <div class="modal-header">
                    <div>
                        <span class="hud-badge" style="background:#ec4899; color:#fff;">FEATURE 1.6 · VIDEO EXPORTER</span>
                        <h2>Export 360° Orbit Showcase Video</h2>
                        <p class="modal-subtitle">Capture smooth high-bitrate WebM/MP4 video clips with camera orbit and continuous token physics for papers, slides, and social sharing.</p>
                    </div>
                    <button id="btn-close-video-modal" class="btn-icon-only modal-close">✕</button>
                </div>

                <div style="padding:20px 24px; display:flex; flex-direction:column; gap:18px;">
                    <div>
                        <label style="font-size:12px; font-weight:700; color:#38bdf8; display:block; margin-bottom:8px;">
                            SELECT PRESET DURATION:
                        </label>
                        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:12px;">
                            <button class="round-button btn-video-preset active" data-duration="5" style="padding:10px; font-size:12px; font-weight:700; background:rgba(56,189,248,0.2); border:1px solid #38bdf8; color:#38bdf8;">
                                🎥 5-Sec Showcase<br><span style="font-size:10px; font-weight:400; color:#94a3b8;">Ideal for Twitter / LinkedIn</span>
                            </button>
                            <button class="round-button btn-video-preset" data-duration="10" style="padding:10px; font-size:12px; font-weight:700; background:#0e1726; border:1px solid rgba(255,255,255,0.1); color:#cbd5e1;">
                                🎬 10-Sec Slow Orbit<br><span style="font-size:10px; font-weight:400; color:#94a3b8;">Ideal for Presentations / Slides</span>
                            </button>
                        </div>
                    </div>

                    <div style="display:flex; justify-content:space-between; align-items:center; font-size:11px; color:#94a3b8; background:#080f1c; padding:10px 14px; border-radius:6px; border:1px solid rgba(255,255,255,0.06);">
                        <span>Framerate: <strong style="color:#38bdf8;">30 FPS (Constant)</strong></span>
                        <span>Bitrate: <strong style="color:#10b981;">8 Mbps (High Quality)</strong></span>
                        <span>Resolution: <strong style="color:#ec4899;">Canvas Native</strong></span>
                    </div>

                    <!-- Progress Box -->
                    <div id="video-progress-box" class="hidden" style="background:#0b1329; border:1px solid rgba(236,72,153,0.3); border-radius:6px; padding:14px;">
                        <div style="display:flex; justify-content:space-between; font-size:11px; margin-bottom:6px;">
                            <span id="video-status-msg" style="font-weight:700; color:#f472b6;">Recording: 0%...</span>
                            <span id="video-status-pct" style="font-family:monospace; color:#cbd5e1;">0%</span>
                        </div>
                        <div style="background:#030712; height:10px; border-radius:5px; overflow:hidden; border:1px solid rgba(255,255,255,0.08);">
                            <div id="video-progress-fill" style="width:0%; height:100%; background:linear-gradient(90deg, #ec4899, #f43f5e); transition:width 0.1s linear;"></div>
                        </div>
                    </div>

                    <div style="display:flex; justify-content:flex-end; gap:10px;">
                        <button id="btn-cancel-video" class="round-button" style="font-size:12px;">Cancel</button>
                        <button id="btn-start-record-video" class="round-button btn-catalog" style="background:#ec4899; color:#fff; font-weight:800; font-size:12px; padding:8px 20px;">
                            Record &amp; Download 🎥
                        </button>
                    </div>
                </div>
            </div>
        `;
        document.body.appendChild(modal);

        const closeBtn = document.getElementById('btn-close-video-modal');
        if (closeBtn) closeBtn.onclick = () => modal.classList.add('hidden');
        const cancelBtn = document.getElementById('btn-cancel-video');
        if (cancelBtn) cancelBtn.onclick = () => modal.classList.add('hidden');
        modal.onclick = (e) => { if (e.target === modal) modal.classList.add('hidden'); };
    }

    let selectedDuration = 5;
    const presetBtns = modal.querySelectorAll('.btn-video-preset');
    presetBtns.forEach(btn => {
        btn.onclick = () => {
            presetBtns.forEach(b => {
                b.style.background = '#0e1726';
                b.style.borderColor = 'rgba(255,255,255,0.1)';
                b.style.color = '#cbd5e1';
            });
            btn.style.background = 'rgba(56,189,248,0.2)';
            btn.style.borderColor = '#38bdf8';
            btn.style.color = '#38bdf8';
            selectedDuration = parseInt(btn.getAttribute('data-duration'), 10);
        };
    });

    const btnStart = document.getElementById('btn-start-record-video');
    const progressBox = document.getElementById('video-progress-box');
    const progressFill = document.getElementById('video-progress-fill');
    const statusMsg = document.getElementById('video-status-msg');
    const statusPct = document.getElementById('video-status-pct');

    if (btnStart) {
        btnStart.onclick = async () => {
            btnStart.disabled = true;
            btnStart.style.opacity = '0.5';
            if (progressBox) progressBox.classList.remove('hidden');

            try {
                await recordCanvasOrbitLoop({
                    canvas,
                    camera,
                    controls,
                    durationSec: selectedDuration,
                    fps: 30,
                    onProgress: ({ stage, percent, message }) => {
                        if (progressFill) progressFill.style.width = `${percent}%`;
                        if (statusPct) statusPct.textContent = `${percent}%`;
                        if (statusMsg) statusMsg.textContent = message;
                    }
                });
                setTimeout(() => {
                    modal.classList.add('hidden');
                    btnStart.disabled = false;
                    btnStart.style.opacity = '1';
                    if (progressBox) progressBox.classList.add('hidden');
                }, 1500);
            } catch (err) {
                if (statusMsg) statusMsg.textContent = `Error: ${err.message}`;
                btnStart.disabled = false;
                btnStart.style.opacity = '1';
            }
        };
    }

    modal.classList.remove('hidden');
}
