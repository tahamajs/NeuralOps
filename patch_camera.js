const fs = require('fs');
let code = fs.readFileSync('Docs/visualizer/app.js', 'utf8');

code = code.replace(/smoothFlyCamera\(8, 22, 0\);/g, `smoothFlyCamera(10, 25, 0); // 3D View`);
code = code.replace(/smoothFlyCamera\(0, 30, 0\);/g, `smoothFlyCamera(15, 40, 0); // 2D Front View\n            // Lock horizontal/vertical angles for 2D mode\n            controls.minAzimuthAngle = 0;\n            controls.maxAzimuthAngle = 0;\n            controls.minPolarAngle = Math.PI / 2;\n            controls.maxPolarAngle = Math.PI / 2;`);

// And when switching back to 3D, restore angles
code = code.replace(/controls\.enableRotate = true;/g, `controls.enableRotate = true;\n            controls.minAzimuthAngle = -Infinity;\n            controls.maxAzimuthAngle = Infinity;\n            controls.minPolarAngle = 0;\n            controls.maxPolarAngle = Math.PI;`);

fs.writeFileSync('Docs/visualizer/app.js', code);
console.log("Patched camera bounds");
