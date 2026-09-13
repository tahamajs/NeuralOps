const fs = require('fs');
let code = fs.readFileSync('Docs/visualizer/style.css', 'utf8');

const additionalCSS = `
/* Pipeline Connectors */
.component-pipeline-list {
    position: relative;
    padding-left: 20px;
}

.component-pipeline-list::before {
    content: '';
    position: absolute;
    top: 15px;
    bottom: 25px;
    left: 8px;
    width: 2px;
    background: linear-gradient(to bottom, #10b981, #38bdf8, #8b5cf6, #f43f5e);
    opacity: 0.5;
    z-index: 1;
}

.component-pipe-item {
    position: relative;
    z-index: 2;
    margin-left: 15px;
}

.component-pipe-item::before {
    content: '';
    position: absolute;
    left: -27px;
    top: 50%;
    width: 12px;
    height: 12px;
    background: var(--bg-surface);
    border: 2px solid var(--cpi-color, #2dd4bf);
    border-radius: 50%;
    transform: translateY(-50%);
    z-index: 3;
    transition: all 0.3s ease;
}

.component-pipe-item.active-highlight::before {
    background: var(--cpi-color, #53ecd9);
    box-shadow: 0 0 10px var(--cpi-color, #53ecd9);
    transform: translateY(-50%) scale(1.3);
}

.component-pipe-item::after {
    content: '';
    position: absolute;
    left: -21px;
    top: 50%;
    width: 21px;
    height: 2px;
    background: rgba(255, 255, 255, 0.2);
    z-index: 1;
}

.component-pipe-item:hover::after {
    background: var(--cpi-color, #53ecd9);
}
`;

if (!code.includes('/* Pipeline Connectors */')) {
    fs.writeFileSync('Docs/visualizer/style.css', code + '\n' + additionalCSS);
    console.log("Patched CSS");
} else {
    console.log("Already patched CSS");
}
