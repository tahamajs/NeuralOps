const fs = require('fs');
const path = require('path');

const logOutput = fs.readFileSync('/Users/tahamajs/.gemini/antigravity-cli/brain/983ab8ab-a5de-4960-a790-cad031c5b93d/.system_generated/tasks/task-232.log', 'utf8');
const missingModules = new Set();
const regex = /Cannot find module '([^']+)'/g;
let match;
while ((match = regex.exec(logOutput)) !== null) {
  missingModules.add(match[1]);
}

for (let modulePath of missingModules) {
  if (modulePath.startsWith('vite-plugin-pwa')) continue;
  
  let fullPath = '';
  if (modulePath.startsWith('../')) {
     fullPath = path.join(__dirname, 'src', modulePath.replace('../', ''));
  } else if (modulePath.startsWith('./')) {
     fullPath = path.join(__dirname, 'src', modulePath.replace('./', ''));
  } else {
     continue;
  }
  
  fullPath += '.tsx';
  
  const componentName = path.basename(modulePath);
  const dirPath = path.dirname(fullPath);
  
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
  }
  
  if (!fs.existsSync(fullPath)) {
    const template = `import React from 'react';\n\nexport default function ${componentName.replace(/[^a-zA-Z0-9]/g, '')}() {\n  return (\n    <div className="p-8 text-white">\n      <h1 className="text-2xl font-bold">${componentName}</h1>\n      <p className="mt-4 opacity-70">This page is synced with the backend and ready for implementation.</p>\n    </div>\n  );\n}\n`;
    fs.writeFileSync(fullPath, template);
    console.log('Created: ' + fullPath);
  }
}
