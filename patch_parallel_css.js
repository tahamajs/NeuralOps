const fs = require('fs');
let css = fs.readFileSync('Docs/visualizer/style.css', 'utf8');

const additionalCSS = `
/* Parallel Pipeline Grouping */
.pipeline-parallel-group {
    display: flex;
    flex-direction: column;
    gap: 4px;
    margin-left: 20px; /* Indent branches */
    padding-left: 15px;
    position: relative;
    border-left: 2px dashed rgba(83, 236, 217, 0.4);
    margin-top: 4px;
    margin-bottom: 4px;
}

.pipeline-parallel-group::before {
    content: 'Branching Dataflow';
    position: absolute;
    top: -12px;
    left: 4px;
    font-size: 8px;
    color: var(--amber);
    letter-spacing: 0.5px;
    background: var(--bg-surface);
    padding: 0 4px;
}

.pipeline-parallel-group .component-pipe-item {
    margin-left: 0;
}

.pipeline-parallel-group .component-pipe-item::after {
    left: -17px;
    width: 17px;
}
.pipeline-parallel-group .component-pipe-item::before {
    left: -23px;
}
`;

if (!css.includes('.pipeline-parallel-group')) {
    fs.writeFileSync('Docs/visualizer/style.css', css + '\\n' + additionalCSS);
    console.log("Patched parallel CSS!");
} else {
    console.log("Parallel CSS already patched!");
}
