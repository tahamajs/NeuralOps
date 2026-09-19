/**
 * ui/keyboard.js
 * ==============
 * Accessibility (a11y) & Full Keyboard-Only Navigation Engine.
 * Supports Arrow-key layer and timeline traversal, ARIA live region announcements,
 * and high-visibility focus indicator rings.
 */

let activeLayerIndex = 0;
let maxLayers = 12;

/**
 * Updates the screen-reader ARIA live polite region.
 */
export function announceLive(message) {
    if (typeof document === 'undefined') return;
    let announcer = document.getElementById('a11y-announcer');
    if (!announcer) {
        announcer = document.createElement('div');
        announcer.id = 'a11y-announcer';
        announcer.setAttribute('role', 'status');
        announcer.setAttribute('aria-live', 'polite');
        announcer.setAttribute('aria-atomic', 'true');
        announcer.style.cssText = 'position:absolute; width:1px; height:1px; margin:-1px; padding:0; overflow:hidden; clip:rect(0,0,0,0); border:0;';
        document.body.appendChild(announcer);
    }
    announcer.textContent = message;
}

/**
 * Sets up global accessibility keyboard shortcuts.
 */
export function setupKeyboardNavigation(handlers = {}) {
    if (typeof window === 'undefined') return;

    window.addEventListener('keydown', (e) => {
        if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA')) return;

        switch (e.key) {
            case 'ArrowUp':
                e.preventDefault();
                activeLayerIndex = Math.min(maxLayers - 1, activeLayerIndex + 1);
                announceLive(`Navigated up to Layer ${activeLayerIndex}`);
                if (handlers.onNavigateLayer) handlers.onNavigateLayer(activeLayerIndex);
                break;

            case 'ArrowDown':
                e.preventDefault();
                activeLayerIndex = Math.max(0, activeLayerIndex - 1);
                announceLive(`Navigated down to Layer ${activeLayerIndex}`);
                if (handlers.onNavigateLayer) handlers.onNavigateLayer(activeLayerIndex);
                break;

            case 'ArrowRight':
                e.preventDefault();
                announceLive('Scrubbed timeline forward');
                if (handlers.onStepForward) handlers.onStepForward();
                break;

            case 'ArrowLeft':
                e.preventDefault();
                announceLive('Scrubbed timeline backward');
                if (handlers.onStepBackward) handlers.onStepBackward();
                break;

            case 'Enter':
                announceLive('Activated selected architecture component');
                if (handlers.onSelectCurrentNode) handlers.onSelectCurrentNode();
                break;
        }
    });
}

export function setActiveLayer(idx, total = 12) {
    activeLayerIndex = idx;
    maxLayers = total;
}

export function getActiveLayer() {
    return activeLayerIndex;
}
