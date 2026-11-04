/**
 * app.js
 * ======
 * 3D Neural Architecture Visualizer Engine — Modular Bootstrap.
 * 
 * Delegates execution to the refactored ES modular architecture in ./src/index.js
 * providing:
 * - Clean separation of concerns (Core, Builders, Physics, UI, Educational)
 * - Semantic Zoom & Continuous Level-of-Detail (LOD 0, LOD 1)
 * - High-Precision Token Physics across 4 Model Families (DistilBERT, RoBERTa, GPT-2, Qwen3.5)
 * - Dual-Viewport Comparative Architecture (Baseline vs Compressed)
 * - Research Module Library (240+ Modules)
 */

import { bootstrapApp } from './src/index.js';

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        bootstrapApp().catch(err => {
            console.error('[Visualizer] Boot failure:', err);
        });
    });
} else {
    bootstrapApp().catch(err => {
        console.error('[Visualizer] Boot failure:', err);
    });
}
