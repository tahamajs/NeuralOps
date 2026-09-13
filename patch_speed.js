const fs = require('fs');
let code = fs.readFileSync('Docs/visualizer/app.js', 'utf8');

code = code.replace(/particleSpeedMultiplier = parseFloat\(e.target.value\);[\s\S]*?\/\//, `simulationSpeed = parseFloat(e.target.value) * 0.015; //`);

fs.writeFileSync('Docs/visualizer/app.js', code);
console.log("Patched speed variable");
