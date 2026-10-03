/**
 * core/palettes.js
 * ================
 * Research-Grade Colorblind-Safe Color Palettes & Contrast Themes.
 * Supports Default Cyan-Violet, Deuteranopia (green-blind), and Protanopia (red-blind).
 */

export const PALETTES = {
    default: {
        id: 'default',
        name: 'Default (Deep Space Cyan)',
        colors: {
            cyan: '#22d5f7',
            violet: '#aa8cff',
            amber: '#ffb34c',
            green: '#10b981',
            rose: '#f43f5e',
            blue: '#3b82f6',
            bg: '#050a12',
            surface: 'rgba(9, 20, 33, 0.72)'
        }
    },
    deuteranopia: {
        id: 'deuteranopia',
        name: 'Deuteranopia-Safe (Blue / Yellow / Amethyst)',
        colors: {
            cyan: '#3b82f6',     // High-contrast blue
            violet: '#a855f7',   // Deep purple
            amber: '#facc15',    // Bright yellow
            green: '#38bdf8',    // Sky blue (replaces green)
            rose: '#dc2626',     // Dark vermilion
            blue: '#2563eb',
            bg: '#050a12',
            surface: 'rgba(9, 20, 33, 0.72)'
        }
    },
    protanopia: {
        id: 'protanopia',
        name: 'Protanopia-Safe (Cerulean / Mint / Amber)',
        colors: {
            cyan: '#0ea5e9',     // Cerulean
            violet: '#818cf8',   // Periwinkle
            amber: '#eab308',    // Saffron amber
            green: '#22c55e',    // Vibrant mint
            rose: '#ec4899',     // Hot pink (replaces red)
            blue: '#0284c7',
            bg: '#050a12',
            surface: 'rgba(9, 20, 33, 0.72)'
        }
    }
};

let activePaletteId = 'default';

export function getActivePalette() {
    return PALETTES[activePaletteId] || PALETTES.default;
}

export function setActivePalette(paletteId) {
    if (!PALETTES[paletteId]) return;
    activePaletteId = paletteId;

    if (typeof document !== 'undefined' && document.documentElement) {
        const pal = PALETTES[paletteId];
        const root = document.documentElement;
        root.style.setProperty('--cyan', pal.colors.cyan);
        root.style.setProperty('--col-svd', pal.colors.cyan);
        root.style.setProperty('--violet', pal.colors.violet);
        root.style.setProperty('--col-moe', pal.colors.violet);
        root.style.setProperty('--amber', pal.colors.amber);
        root.style.setProperty('--col-mla', pal.colors.amber);
        root.style.setProperty('--green', pal.colors.green);
        root.style.setProperty('--rose', pal.colors.rose);
    }
}
