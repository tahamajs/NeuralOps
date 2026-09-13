const fs = require('fs');
let html = fs.readFileSync('Docs/visualizer/index.html', 'utf8');

const ts = Date.now();
html = html.replace(/src="app\.js(\?v=\d+)?"/, \`src="app.js?v=\${ts}"\`);
html = html.replace(/href="style\.css(\?v=\d+)?"/, \`href="style.css?v=\${ts}"\`);

fs.writeFileSync('Docs/visualizer/index.html', html);
console.log("Added cache buster!");
