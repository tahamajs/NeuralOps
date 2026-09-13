const fs = require('fs');
let code = fs.readFileSync('Docs/visualizer/app.js', 'utf8');

const svgGuidesFunc = `
// SVG Leader Lines (3D to HTML Pipeline)
function updateScreenSpaceGuides() {
    const svgEl = document.getElementById('label-guides');
    const pipeItems = document.querySelectorAll('.component-pipe-item');
    if (!svgEl || pipeItems.length === 0) return;

    // We will dynamically create paths
    let pathsHTML = '';
    const width = window.innerWidth;
    const height = window.innerHeight;

    // Ensure SVG covers the screen
    svgEl.style.position = 'absolute';
    svgEl.style.top = '0';
    svgEl.style.left = '0';
    svgEl.style.width = '100%';
    svgEl.style.height = '100%';
    svgEl.style.pointerEvents = 'none';
    svgEl.style.zIndex = '10';

    pipeItems.forEach(item => {
        const compId = item.getAttribute('data-comp-id');
        if (!compId) return;
        
        let targetKey = compId.toLowerCase();
        let targetMesh = interactableMeshes.find(m => {
            const u = m.userData;
            if (!u || u.side !== 'baseline') return false;
            return (u.id && u.id.toLowerCase().includes(targetKey)) ||
                   (u.sub_type && u.sub_type.toLowerCase().includes(targetKey)) ||
                   (u.type && u.type.toLowerCase().includes(targetKey)) ||
                   (u.label && u.label.toLowerCase().includes(targetKey));
        });
        
        // If no direct match, try a looser match or skip
        if (!targetMesh) return;

        // Project 3D position to 2D screen space
        const pos3D = targetMesh.position.clone();
        pos3D.project(camera);
        
        // Check if behind camera
        if (pos3D.z > 1) return;

        const screenX = (pos3D.x *  .5 + .5) * width;
        const screenY = (pos3D.y * -.5 + .5) * height;

        // Get target HTML element position
        const rect = item.getBoundingClientRect();
        const targetX = rect.left;
        const targetY = rect.top + rect.height / 2;

        // Extract color from CSS variable or default
        const color = item.style.getPropertyValue('--cpi-color') || '#2dd4bf';

        // Draw curved SVG path
        // M: move to 3D point, C: cubic bezier curve to HTML element
        const cp1x = screenX + (targetX - screenX) * 0.4;
        const cp1y = screenY;
        const cp2x = screenX + (targetX - screenX) * 0.6;
        const cp2y = targetY;
        
        pathsHTML += \`<path d="M \${screenX} \${screenY} C \${cp1x} \${cp1y}, \${cp2x} \${cp2y}, \${targetX} \${targetY}" fill="none" stroke="\${color}" stroke-width="1.5" stroke-dasharray="4 4" opacity="0.6"></path>\`;
        
        // Add a dot at the 3D mesh location
        pathsHTML += \`<circle cx="\${screenX}" cy="\${screenY}" r="3" fill="\${color}" opacity="0.8" />\`;
    });

    svgEl.innerHTML = pathsHTML;
}
`;

// Insert the function before `function render()`
const renderIdx = code.indexOf('function render(time) {');
if (renderIdx !== -1) {
    code = code.substring(0, renderIdx) + svgGuidesFunc + '\n' + code.substring(renderIdx);
    
    // Call it inside render(time)
    const renderCallIdx = code.indexOf('updateTokens(tokenMeshCompressed, trailMeshCompressed, tokenDataCompressed, true);');
    if (renderCallIdx !== -1) {
        code = code.substring(0, renderCallIdx + 82) + '\n    updateScreenSpaceGuides();' + code.substring(renderCallIdx + 82);
        fs.writeFileSync('Docs/visualizer/app.js', code);
        console.log("Patched SVG guides!");
    } else {
        console.log("Could not find renderCallIdx");
    }
} else {
    console.log("Could not find renderIdx");
}
