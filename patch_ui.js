const fs = require('fs');
let code = fs.readFileSync('Docs/visualizer/app.js', 'utf8');

const uiLogic = `
// ---------------------------------------------------------------------------
// 22. Polymetis / Attention Lab UI Wiring
// ---------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
    // 2D/3D View Toggle
    const btn3d = document.getElementById('btn-view-3d');
    const btn2d = document.getElementById('btn-view-2d');
    if (btn3d && btn2d) {
        btn3d.addEventListener('click', () => {
            btn3d.classList.add('active');
            btn2d.classList.remove('active');
            btn3d.setAttribute('aria-selected', 'true');
            btn2d.setAttribute('aria-selected', 'false');
            
            // Switch to 3D perspective
            controls.enableRotate = true;
            controls.enableZoom = true;
            smoothFlyCamera(8, 22, 0); // Restore typical 3D angle
        });
        
        btn2d.addEventListener('click', () => {
            btn2d.classList.add('active');
            btn3d.classList.remove('active');
            btn2d.setAttribute('aria-selected', 'true');
            btn3d.setAttribute('aria-selected', 'false');
            
            // Switch to 2D Orthographic-like locked perspective
            controls.enableRotate = false;
            // Align camera flat against the network
            smoothFlyCamera(0, 30, 0);
        });
    }

    // Mechanism Navigation Buttons
    const mechButtons = document.querySelectorAll('.mechanism-button');
    mechButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            mechButtons.forEach(b => {
                b.classList.remove('active');
                b.setAttribute('aria-selected', 'false');
            });
            btn.classList.add('active');
            btn.setAttribute('aria-selected', 'true');
            
            const mModel = btn.dataset.model;
            const mModule = btn.dataset.module || currentCompression;
            if (mModel) {
                if (window.switchModelFromMatrix) {
                    window.switchModelFromMatrix(mModel);
                } else {
                    currentModel = mModel;
                    currentCompression = mModule;
                    loadModelAndModule(currentModel, currentCompression);
                }
            }
        });
    });

    // Control Dock (Play/Pause, Speed, Scrubber)
    const btnPlay = document.getElementById('btn-play-pause');
    const selSpeed = document.getElementById('select-speed');
    const scrubber = document.getElementById('timeline-scrubber');
    const timeCode = document.getElementById('timeline-step');
    
    if (btnPlay) {
        btnPlay.addEventListener('click', () => {
            isParticlePaused = !isParticlePaused; // Make sure this global variable exists
            btnPlay.innerText = isParticlePaused ? 'Play' : 'Pause';
            btnPlay.classList.toggle('paused', isParticlePaused);
        });
    }
    
    if (selSpeed) {
        selSpeed.addEventListener('change', (e) => {
            particleSpeedMultiplier = parseFloat(e.target.value); // Assume this controls speed
        });
    }
    
    if (scrubber && timeCode) {
        scrubber.addEventListener('input', (e) => {
            const step = parseInt(e.target.value, 10);
            timeCode.innerText = \`Step \${step + 1}/7\`;
            // Optional: Snap the camera to specific components based on the step
            const stages = [
                'input_embeddings',
                'q_proj',
                'attn.residual_add',
                'gate_proj',
                'ffn.residual_add',
                'output_head'
            ];
            if (stages[step] && window.focusComponentIn3D) {
                window.focusComponentIn3D(stages[step]);
            }
        });
    }
});
`;

if (!code.includes('22. Polymetis / Attention Lab UI Wiring')) {
    fs.writeFileSync('Docs/visualizer/app.js', code + '\n' + uiLogic);
    console.log("Patched UI wiring");
} else {
    console.log("Already patched UI wiring");
}
