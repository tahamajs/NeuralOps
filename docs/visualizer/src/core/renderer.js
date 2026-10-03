/**
 * core/renderer.js
 * ================
 * WebGLRenderer initialization, tone mapping, and responsive resize handling.
 */

import * as THREE from 'three';

/**
 * Creates and configures the Three.js WebGLRenderer with fallback detection.
 * @param {HTMLCanvasElement} canvas - Target canvas
 * @returns {THREE.WebGLRenderer} Configured renderer instance
 */
export function createRenderer(canvas) {
    // Check WebGL availability
    const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
    if (!gl) {
        const fallbackBanner = document.getElementById('webgl-fallback');
        if (fallbackBanner) fallbackBanner.style.display = 'block';
        throw new Error('WebGL is not supported in this environment');
    }

    const renderer = new THREE.WebGLRenderer({
        canvas,
        antialias: true,
        alpha: true,
        powerPreference: 'high-performance'
    });

    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2.0));
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setScissorTest(true);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.3;

    return renderer;
}

/**
 * Binds window resize events to the renderer and cameras.
 * @param {THREE.WebGLRenderer} renderer - Active renderer
 * @param {THREE.PerspectiveCamera} camera - Active camera
 */
export function setupResizeHandler(renderer, camera) {
    window.addEventListener('resize', () => {
        const width = window.innerWidth;
        const height = window.innerHeight;
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        renderer.setSize(width, height);
    });
}
