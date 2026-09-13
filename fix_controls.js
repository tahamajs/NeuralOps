const fs = require('fs');
let code = fs.readFileSync('Docs/visualizer/app.js', 'utf8');

// Fix Speed
code = code.replace(/simulationSpeed = parseFloat\\(e.target.value\\) \\* 0\\.015;/g, 'simulationSpeed = parseFloat(e.target.value);');

// Remove duplicate play/pause event listener
code = code.replace(/const btnPlay = document\.getElementById\\('btn-play-pause'\\);[\\s\\S]*?if \\(selSpeed\\) {/, 'const selSpeed = document.getElementById(\\'select-speed\\');\\n    const scrubber = document.getElementById(\\'timeline-scrubber\\');\\n    const timeCode = document.getElementById(\\'timeline-step\\');\\n\\n    if (selSpeed) {');

fs.writeFileSync('Docs/visualizer/app.js', code);
console.log("Fixed controls");
